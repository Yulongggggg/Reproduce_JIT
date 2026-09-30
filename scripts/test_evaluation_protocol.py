"""CPU regression tests: formal comparison eligibility and interrupted search recovery.

Synthetic scores here test orchestration only; they are never written into reports/runs.
"""
import copy
import json
from pathlib import Path
import unittest
import tempfile
from unittest.mock import patch

from evaluation_protocol import comparison, protocol_for, run_final_evaluations

ROOT = Path(__file__).resolve().parents[1]


class Evaluator:
    def __init__(self, config):
        self.config = config
        self.cache = {}
        self.stop_at = None

    def __call__(self, ema, cfg, count, phase, epoch):
        key = (ema, cfg, count, phase, epoch)
        if key not in self.cache:
            if len(self.cache) == self.stop_at:
                raise TimeoutError('Simulated allocation timeout')
            self.cache[key] = {
                'model': self.config['model'], 'resolution': self.config['img_size'],
                'completed_epochs': epoch, 'num_images': count,
                'samples_per_class': count // 1000,
                'evaluation_protocol': protocol_for(self.config), 'ema': ema, 'cfg': cfg,
                'frechet_inception_distance': 10 + (cfg - 3.6)**2 + (ema - 0.9996)*1000,
            }
        return copy.deepcopy(self.cache[key])


class EvaluationProtocolTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT/'configs/b16_4gpu_200ep.json').read_text())
        self.evaluator = Evaluator(self.config)
        self.saved = []
        self.selections = []

    def run_pipeline(self):
        return run_final_evaluations(self.config, self.evaluator,
            lambda value: self.saved.append(copy.deepcopy(value)),
            lambda value: self.selections.append(copy.deepcopy(value)))

    def test_full_search_keeps_fixed_cfg_and_uses_50k_for_both_finals(self):
        summary = self.run_pipeline()
        self.assertEqual(summary['final_50k_official_cfg'][0]['cfg'], 2.9)
        final = summary['final_50k_paper_protocol']
        self.assertEqual((final['ema'], final['cfg'], final['num_images']), (0.9996, 3.6, 50000))
        self.assertEqual(self.selections[-1]['completed_candidates'], 93)
        self.assertEqual(summary['paper_comparison']['status'], 'matched')
        self.assertEqual(summary['paper_comparison']['paper_fid'], 4.37)
        # Reuse fixed-CFG selection points and the identical issue #56 final.
        self.assertEqual(len(self.evaluator.cache), 93 + 2)
        self.assertEqual([s['status'] for s in self.saved], ['evaluating', 'evaluating', 'complete'])

    def test_allocation_timeout_resumes_without_recomputing_completed_points(self):
        self.evaluator.stop_at = 22
        with self.assertRaises(TimeoutError):
            self.run_pipeline()
        durable = copy.deepcopy(self.evaluator.cache)
        self.assertTrue(all(s['status'] == 'evaluating' for s in self.saved))
        self.evaluator.stop_at = None
        summary = self.run_pipeline()
        self.assertEqual(summary['status'], 'complete')
        self.assertEqual(len(self.evaluator.cache), 95)
        self.assertTrue(all(self.evaluator.cache[k] == v for k, v in durable.items()))

    def test_wrong_epoch_count_sampler_reference_or_selection_cannot_get_paper_delta(self):
        good = self.run_pipeline()
        for field, bad in [('num_images', 8000), ('completed_epochs', 80),
                           ('resolution', 512), ('cfg', 2.9), ('frechet_inception_distance', float('nan'))]:
            with self.subTest(field=field):
                summary = copy.deepcopy(good)
                summary['final_50k_paper_protocol'][field] = bad
                row = comparison(summary, self.config)
                self.assertEqual(row['status'], 'ineligible')
                self.assertIsNone(row['delta'])
        for field, bad in [('num_sampling_steps', 250), ('fid_statistics_sha256', 'different')]:
            with self.subTest(field=field):
                summary = copy.deepcopy(good)
                summary['final_50k_paper_protocol']['evaluation_protocol'][field] = bad
                self.assertEqual(comparison(summary, self.config)['status'], 'ineligible')
        summary = copy.deepcopy(good)
        summary['paper_selection_on_8k']['candidates'].pop()
        self.assertEqual(comparison(summary, self.config)['status'], 'ineligible')
        summary = copy.deepcopy(good)
        summary['paper_selection_on_8k']['candidates'][0]['completed_epochs'] = 100
        self.assertEqual(comparison(summary, self.config)['status'], 'ineligible')
        self.assertIsNone(comparison(good, {**self.config, 'epochs': 80})['paper_fid'])

    def test_fixed_cfg_alone_is_not_claimed_as_full_paper_protocol(self):
        self.run_pipeline()
        row = comparison(self.saved[0], self.config)
        self.assertEqual(row['status'], 'pending')
        self.assertIsNone(row['our_fid'])
        self.assertIsNone(row['delta'])

    def test_l16_preserves_its_own_cfg_and_reference(self):
        self.config = json.loads((ROOT/'configs/l16_4gpu_200ep.json').read_text())
        self.evaluator = Evaluator(self.config)
        summary = self.run_pipeline()
        self.assertEqual(summary['final_50k_official_cfg'][0]['cfg'], 2.4)
        self.assertIsNone(summary['issue56_diagnostic_50k'])
        self.assertEqual(summary['paper_comparison']['paper_fid'], 2.79)

    def test_report_keeps_8k_out_of_formal_table_and_tracks_partial_50k(self):
        import report
        self.run_pipeline()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'configs').mkdir()
            (root/'reports').mkdir()
            for name in ('b16', 'l16'):
                config = json.loads((ROOT/f'configs/{name}_4gpu_200ep.json').read_text())
                (root/f'configs/{name}_4gpu_200ep.json').write_text(json.dumps(config))
                config8 = json.loads((ROOT/f'configs/{name}_8gpu_standard_200ep.json').read_text())
                (root/f'configs/{name}_8gpu_standard_200ep.json').write_text(json.dumps(config8))
            (root/'reports/experiment_plan.json').write_text('{"standard8_enabled":true}')
            run = root/'runs/b16_4gpu_200ep'
            (run/'evaluations').mkdir(parents=True)
            (run/'summary.json').write_text(json.dumps(self.saved[0]))
            (run/'progress.json').write_text(json.dumps({'completed_epochs': 200}))
            (run/'run_metadata.json').write_text('{"world_size":8}')
            (run/'evaluations/monitor-ep080.json').write_text(json.dumps({
                'completed_epochs': 80, 'num_images': 8000, 'cfg': 2.9,
                'ema': 0.9996, 'frechet_inception_distance': 123.456}))
            with patch.object(report, 'ROOT', root), patch.object(report, 'REPORTS', root/'reports'):
                report.main()
            document = (root/'reports/REPORT.md').read_text()
            self.assertNotIn('|\n\n|', document)
            formal = document.split('## 同 epoch')[1].split('### README')[0]
            self.assertNotIn('123.456', formal)
            self.assertIn('| JiT-B/16 | 200 | 256² | 4.37 | 待评估 |', formal)
            self.assertFalse(json.loads((root/'reports/status.json').read_text())['complete'])
            rows = json.loads((root/'reports/comparison_50k.json').read_text())
            self.assertEqual(rows[0]['status'], 'pending')
            self.assertEqual(rows[0]['gpu_count'], 8)
            self.assertIsNone(rows[0]['delta'])


if __name__ == '__main__':
    unittest.main()
