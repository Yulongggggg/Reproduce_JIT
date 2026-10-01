"""Check concurrency and continuation safety without allocating any GPUs."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import resource_queue as queue


class ResourceQueueTests(unittest.TestCase):
    def test_model_lock_excludes_other_process_and_releases(self):
        with tempfile.TemporaryDirectory() as directory:
            handle = queue.try_model_lock(directory)
            command = [sys.executable, '-c',
                       'import sys; from resource_queue import try_model_lock; '
                       'h=try_model_lock(sys.argv[1]); sys.exit(0 if h else 75)', directory]
            self.assertEqual(subprocess.run(command, cwd=Path(__file__).parent).returncode, 75)
            handle.close()
            self.assertEqual(subprocess.run(command, cwd=Path(__file__).parent).returncode, 0)

    def test_both_shapes_keep_four_gpus_and_forty_cpu_cores(self):
        for profile in ('single', 'split'):
            nodes, gpus, cpus, _ = queue.PROFILES[profile]
            self.assertEqual(nodes*gpus, 4)
            self.assertEqual(nodes*cpus, 40)
            command = queue.sbatch_command('l16', profile, 1234)
            self.assertIn('--dependency=afternotok:1234', command)
            self.assertIn('--time-min=03:00:00', command)
            self.assertEqual(command[-2:], ['l16', profile])

    def test_standard_requests_eight_gpus_and_24_to_48_hours(self):
        command = queue.sbatch_command('b16', 'standard8', 1234)
        for flag in ('--qos=standard', '--nodes=1', '--gres=gpu:8',
                     '--time=2-00:00:00', '--time-min=1-00:00:00',
                     '--dependency=afternotok:1234'):
            self.assertIn(flag, command)
        self.assertEqual(command[-2:], ['b16', 'standard8'])

    def test_long_requests_eight_gpus_and_48_hours_to_seven_days(self):
        command = queue.sbatch_command('l16', 'long8', 1234)
        for flag in ('--qos=long', '--nodes=1', '--gres=gpu:8',
                     '--time=7-00:00:00', '--time-min=2-00:00:00',
                     '--dependency=afternotok:1234'):
            self.assertIn(flag, command)
        self.assertEqual(command[-2:], ['l16', 'long8'])

    def test_disabled_profiles_cannot_submit_or_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'reports').mkdir()
            with patch.object(queue, 'ROOT', root), \
                 patch.object(queue.subprocess, 'check_output') as check, \
                 patch.object(queue.subprocess, 'run') as run:
                for enabled in ('long8', 'standard8'):
                    (root/'reports/experiment_plan.json').write_text(json.dumps({'enabled_profiles':[enabled]}))
                    for profile in set(queue.PROFILES) - {enabled}:
                        with self.assertRaises(ValueError):
                            queue.submit('b16', profile)
                        self.assertEqual(queue.worker('b16', profile), 0)
                check.assert_not_called()
                run.assert_not_called()

    def test_duplicate_submission_reuses_existing_job(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = root/'jobs.json'
            record.write_text(json.dumps({'jobs': [
                {'model': 'l16', 'profile': 'single', 'job_id': 42}]}))
            with patch.object(queue, 'ROOT', root), patch.object(queue, 'RECORD', record), \
                 patch.object(queue.subprocess, 'check_output', return_value='42|PENDING\n') as call:
                self.assertEqual(queue.submit('l16', 'single'), 42)
                self.assertEqual(call.call_count, 1)
                self.assertEqual(call.call_args.args[0][0], 'squeue')

    def test_split_launcher_uses_tcp_on_both_nodes_and_preserves_four_ranks(self):
        # Execute the real shell entrypoint without requesting GPUs or importing CUDA.
        with tempfile.TemporaryDirectory() as directory:
            launcher = Path(directory)/'torchrun'
            launcher.write_text(f'#!{sys.executable}\n'
                'import json, os, sys\n'
                'keys = ("NCCL_NET", "NCCL_IB_DISABLE", "JIT_NCCL_PREFLIGHT", "NCCL_P2P_DISABLE")\n'
                'print(json.dumps({"argv": sys.argv[1:], "env": {k: os.environ.get(k) for k in keys}}))\n')
            launcher.chmod(0o755)
            for node in (0, 1):
                environment = dict(os.environ, PATH=directory+os.pathsep+os.environ['PATH'],
                    SLURM_NODEID=str(node), JIT_MASTER_ADDR='example-node', JIT_MASTER_PORT='23456',
                    NCCL_NET='IB', NCCL_IB_DISABLE='0', NCCL_P2P_DISABLE='0')
                result = json.loads(subprocess.check_output(
                    ['bash', str(Path(__file__).with_name('torchrun_node.sh')), 'config.json'],
                    env=environment, text=True))
                self.assertEqual(result['env'], {'NCCL_NET': 'Socket', 'NCCL_IB_DISABLE': '1',
                    'JIT_NCCL_PREFLIGHT': '1', 'NCCL_P2P_DISABLE': '0'})
                self.assertEqual(result['argv'], ['--nnodes=2', '--nproc_per_node=2',
                    f'--node_rank={node}', '--master_addr=example-node', '--master_port=23456',
                    'scripts/run.py', '--mode', 'train', '--config', 'config.json'])

    def prepare(self, directory):
        root = Path(directory)
        (root/'configs').mkdir()
        (root/'configs/l16_4gpu_200ep.json').write_text(json.dumps({'output_dir': 'runs/l16'}))
        return root, root/'runs/l16'

    def test_busy_candidate_releases_without_training_or_successor(self):
        with tempfile.TemporaryDirectory() as directory:
            root, output = self.prepare(directory)
            with queue.try_model_lock(output), patch.object(queue, 'ROOT', root), \
                 patch.object(queue, 'submit') as submit, \
                 patch.object(queue.subprocess, 'run') as run:
                self.assertEqual(queue.worker('l16', 'single'), 0)
                submit.assert_not_called()
                run.assert_not_called()

    def test_four_gpu_worker_skips_after_eight_gpu_takeover(self):
        with tempfile.TemporaryDirectory() as directory:
            root, output = self.prepare(directory)
            output.mkdir(parents=True)
            (output/'execution_world.json').write_text('{"world_size":8}')
            with patch.object(queue, 'ROOT', root), patch.object(queue, 'submit') as submit, \
                 patch.object(queue.subprocess, 'run') as run:
                self.assertEqual(queue.worker('l16', 'single'), 0)
                submit.assert_not_called()
                run.assert_not_called()

    def check_eight_gpu_worker(self, profile):
        with tempfile.TemporaryDirectory() as directory:
            root, output = self.prepare(directory)
            (root/'configs/l16_8gpu_standard_200ep.json').write_text(json.dumps({'output_dir': 'runs/l16'}))
            def run(command, **kwargs):
                self.assertIsNone(queue.try_model_lock(output))
                if command[0] == 'torchrun':
                    self.assertIn('--nproc_per_node=8', command)
                    self.assertIn('--allow-expand-to-eight', command)
                    self.assertEqual(kwargs['env']['JIT_ALLOCATION_LOCK_HELD'], '1')
                    (output/'summary.json').write_text('{"status":"complete"}')
                return subprocess.CompletedProcess(command, 0)
            with patch.object(queue, 'ROOT', root), patch.object(queue, 'submit', return_value=5678), \
                 patch.dict(os.environ, {'SLURM_JOB_ID':'1234'}), \
                 patch.object(queue.subprocess, 'run', side_effect=run):
                self.assertEqual(queue.worker('l16', profile), 0)

    def test_standard_worker_shares_model_lock_and_enables_expansion(self):
        self.check_eight_gpu_worker('standard8')

    def test_long_worker_shares_model_lock_and_enables_expansion(self):
        self.check_eight_gpu_worker('long8')

    def test_timeout_continues_but_three_crashes_stop(self):
        jobs = {'jobs': [{'job_id': i, 'predecessor': i-1} for i in range(2, 5)]}
        with patch.object(queue, 'read', return_value=jobs):
            with patch.object(queue.subprocess, 'check_output', return_value='1|FAILED|\n2|FAILED|\n3|OUT_OF_MEMORY|\n'):
                self.assertTrue(queue.repeated_failures(4))
            with patch.object(queue.subprocess, 'check_output', return_value='1|FAILED|\n2|TIMEOUT|\n3|FAILED|\n'):
                self.assertFalse(queue.repeated_failures(4))

    def test_successor_precedes_training_and_only_it_is_cancelled_on_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            root, output = self.prepare(directory)
            calls = []

            def submit(model, profile, dependency):
                self.assertEqual(dependency, 1234)
                calls.append('submit')
                return 5678

            def run(command, **kwargs):
                self.assertIsNone(queue.try_model_lock(output))
                if command[0] == 'torchrun':
                    self.assertEqual(calls, ['submit'])
                    self.assertEqual(kwargs['env']['JIT_ALLOCATION_LOCK_HELD'], '1')
                    (output/'summary.json').write_text('{"status":"complete"}')
                    calls.append('train')
                else:
                    self.assertEqual(command, ['scancel', '5678'])
                    calls.append('cancel')
                return subprocess.CompletedProcess(command, 0)

            with patch.object(queue, 'ROOT', root), patch.object(queue, 'submit', side_effect=submit), \
                 patch.dict(os.environ, {'SLURM_JOB_ID': '1234'}), \
                 patch.object(queue.subprocess, 'run', side_effect=run):
                self.assertEqual(queue.worker('l16', 'single'), 0)
            self.assertEqual(calls, ['submit', 'train', 'cancel'])
            with queue.try_model_lock(output):
                pass


if __name__ == '__main__':
    unittest.main()
