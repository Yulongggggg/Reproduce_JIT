"""Checkpoint compatibility checks; these tests allocate no GPUs."""
import copy
import json
from pathlib import Path
import unittest

from resume_policy import validate_resume, expansion_seed

ROOT = Path(__file__).resolve().parents[1]


class ResumePolicyTests(unittest.TestCase):
    def checkpoint(self, model):
        old = json.loads((ROOT/f'configs/{model}_4gpu_200ep.json').read_text())
        target = json.loads((ROOT/f'configs/{model}_8gpu_standard_200ep.json').read_text())
        checkpoint = {'config':old, 'world_size':4, 'rng_per_rank':[{'rank':r} for r in range(4)],
                      'epoch':183, 'global_step':230184, 'model':{'weight':42},
                      'optimizer':{'step':230184}, 'ema_0.9996':{'weight':40},
                      'ema_0.9998':{'weight':39}, 'ema_0.9999':{'weight':38}}
        return checkpoint, target

    def test_expansion_is_explicit_and_preserves_the_entire_checkpoint(self):
        for model in ('b16', 'l16'):
            checkpoint, target = self.checkpoint(model)
            before = copy.deepcopy(checkpoint)
            with self.assertRaises(ValueError):
                validate_resume(checkpoint, target, 8)
            self.assertTrue(validate_resume(checkpoint, target, 8, True))
            self.assertEqual(checkpoint, before)
            # Both samplers consume the same number of optimizer steps per epoch.
            for world, cfg in ((4, checkpoint['config']), (8, target)):
                per_rank = (1281167+world-1)//world
                microsteps = per_rank//cfg['batch_size']
                self.assertEqual(microsteps % cfg['grad_accumulation'], 0)
                self.assertEqual(microsteps//cfg['grad_accumulation'], 1251)

    def test_hyperparameter_drift_batch_change_and_gpu_shrink_are_rejected(self):
        checkpoint, target = self.checkpoint('b16')
        for key, value in [('lr', 0.0001), ('cfg', 3.6), ('epochs', 600),
                           ('grad_accumulation', 2), ('model', 'JiT-L/16')]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_resume(checkpoint, {**target, key:value}, 8, True)
        checkpoint['world_size'] = 8
        checkpoint['config'] = target
        checkpoint['rng_per_rank'] *= 2
        four = {**target, 'grad_accumulation':2}
        with self.assertRaises(ValueError):
            validate_resume(checkpoint, four, 4, True)

    def test_same_world_resumes_normally_and_missing_rng_is_rejected(self):
        checkpoint, _ = self.checkpoint('b16')
        self.assertFalse(validate_resume(checkpoint, checkpoint['config'], 4))
        checkpoint['rng_per_rank'].pop()
        with self.assertRaises(ValueError):
            validate_resume(checkpoint, checkpoint['config'], 4)

    def test_new_rank_seeds_are_distinct_and_reproducible(self):
        seeds = [expansion_seed(0, 184, rank) for rank in range(4, 8)]
        self.assertEqual(len(set(seeds)), 4)
        self.assertEqual(seeds, [expansion_seed(0, 184, rank) for rank in range(4, 8)])
        self.assertNotEqual(seeds[0], expansion_seed(0, 185, 4))


if __name__ == '__main__':
    unittest.main()
