"""Shared checkpoint exclusion and a bounded set of alternative allocations."""
import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / 'reports/jobs_flexible.json'
PROFILES = {'single': (1, 4, 40, '256G'), 'split': (2, 2, 20, '128G'),
            'standard8': (1, 8, 72, '512G'), 'long8': (1, 8, 72, '512G')}


def profile_enabled(profile):
    plan = read(ROOT / 'reports/experiment_plan.json', {})
    return profile in plan.get('enabled_profiles', PROFILES)


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def try_model_lock(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    handle = (output / '.allocation.lock').open('a')
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    return handle


def queue_ids(legacy, record_path=None):
    extra = read(record_path if record_path is not None else RECORD, {}).get('jobs', [])
    return list(dict.fromkeys([*legacy.get('job_ids', []),
                              *(int(job['job_id']) for job in extra)]))


def sbatch_command(model, profile, dependency=None):
    if model not in ('b16', 'l16') or profile not in PROFILES:
        raise ValueError('Unsupported model or allocation profile')
    nodes, gpus, cpus, memory = PROFILES[profile]
    standard8 = profile == 'standard8'
    long8 = profile == 'long8'
    command = ['sbatch', '--parsable', f'--job-name=jit_{model}_{profile}',
               '--partition=alpha', '--account=co_carson_aiaided',
               '--qos=long' if long8 else ('--qos=standard' if standard8 else '--qos=priority'),
               f'--nodes={nodes}', f'--ntasks={nodes}', '--ntasks-per-node=1',
               f'--cpus-per-task={cpus}', f'--gres=gpu:{gpus}',
               '--constraint=h100|h200', f'--mem={memory}',
               '--time=7-00:00:00' if long8 else ('--time=2-00:00:00' if standard8 else '--time=12:00:00'),
               '--time-min=2-00:00:00' if long8 else ('--time-min=1-00:00:00' if standard8 else
               ('--time-min=02:00:00' if model == 'b16' else '--time-min=03:00:00')),
               '--output=logs/flexible-%j.log',
               '--chdir=' + str(ROOT)]
    if dependency:
        command.append(f'--dependency=afternotok:{int(dependency)}')
    return [*command, 'scripts/train_flexible.sbatch', model, profile]


def repeated_failures(job_id):
    """Stop a deterministic crash loop; normal wall-time resumptions are unlimited."""
    jobs = {job['job_id']: job for job in read(RECORD, {}).get('jobs', [])}
    parents = []
    for _ in range(3):
        job_id = jobs.get(job_id, {}).get('predecessor')
        if job_id is None:
            return False
        parents.append(job_id)
    output = subprocess.check_output(['sacct', '-X', '-n', '-P', '-j',
        ','.join(map(str, parents)), '--format=JobIDRaw,State'], text=True, timeout=30)
    states = {parts[0]: parts[1] for line in output.splitlines()
              if len(parts := line.split('|')) >= 2}
    return all(states.get(str(parent), '').split()[0:1] in (['FAILED'], ['OUT_OF_MEMORY'])
               for parent in parents)


def submit(model, profile, dependency=None):
    """One ready candidate per shape; running workers create only one successor."""
    lock_path = ROOT / 'artifacts/flexible_submission.lock'
    lock_path.parent.mkdir(exist_ok=True)
    with lock_path.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        if not profile_enabled(profile):
            raise ValueError(f'Allocation profile {profile} disabled by experiment_plan.json')
        record = read(RECORD, {'purpose': 'Four-GPU candidates sharing per-model checkpoints',
                              'global_batch': 1024, 'jobs': []})
        candidates = [job for job in record['jobs'] if job['model'] == model
                      and job['profile'] == profile
                      and (dependency is None or job.get('predecessor') == dependency)]
        if candidates:
            candidate_ids = {int(job['job_id']) for job in candidates}
            # Old completed job IDs may already be purged from the controller.
            active = subprocess.check_output(['squeue', '-h', '-u', str(os.getuid()), '-o', '%i|%T'],
                                             text=True, timeout=30)
            for line in active.splitlines():
                job_id, state = line.split('|')
                if job_id.isdigit() and int(job_id) in candidate_ids and state in ('PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING'):
                    print(f'Already queued: {model}/{profile} job {job_id}', flush=True)
                    return int(job_id)
        command = sbatch_command(model, profile, dependency)
        job_id = int(subprocess.check_output(command, cwd=ROOT, text=True,
                                             timeout=60).strip().split(';')[0])
        record['jobs'].append({'job_id': job_id, 'model': model, 'profile': profile,
                              'predecessor': dependency,
                              'gpu_count': PROFILES[profile][0]*PROFILES[profile][1],
                              'qos': 'long' if profile == 'long8' else ('standard' if profile == 'standard8' else 'priority'),
                              'nodes': PROFILES[profile][0],
                              'submitted_utc': dt.datetime.now(dt.timezone.utc).isoformat()})
        if profile == 'standard8':
            record['purpose'] = 'Eight-GPU standard candidates sharing existing per-model checkpoints; enabled profiles are controlled by experiment_plan.json'
        elif profile == 'long8':
            record['purpose'] = 'Eight-GPU long jobs sharing existing per-model checkpoints; older candidates retained as history only'
        temporary = RECORD.with_suffix('.tmp')
        temporary.write_text(json.dumps(record, indent=2) + '\n')
        temporary.replace(RECORD)
        print(f'Submitted {model}/{profile}: {job_id}', flush=True)
        return job_id


def worker(model, profile):
    if not profile_enabled(profile):
        print(f'{profile} disabled by experiment plan; releasing allocation.', flush=True)
        return 0
    # QoS changes the allocation, not the eight-rank training configuration.
    suffix = '8gpu_standard_200ep' if profile in ('standard8', 'long8') else '4gpu_200ep'
    config = ROOT / f'configs/{model}_{suffix}.json'
    cfg = read(config)
    output = ROOT / cfg['output_dir']
    allocation_lock = try_model_lock(output)
    if allocation_lock is None:
        print(f'{model} already has a running allocation; releasing these GPUs.', flush=True)
        return 0
    # Keep allocation_lock alive until torchrun and all its children have exited.
    with allocation_lock:
        expected_world = PROFILES[profile][0]*PROFILES[profile][1]
        route = read(output/'execution_world.json', {})
        if route.get('world_size', expected_world) != expected_world:
            print(f'{model} already transferred to {route["world_size"]} GPUs; releasing allocation.', flush=True)
            return 0
        if read(output/'summary.json', {}).get('status') == 'complete':
            print(f'{model} already complete; releasing allocation.', flush=True)
            return 0
        job_id = int(os.environ['SLURM_JOB_ID'])
        if repeated_failures(job_id):
            error = {'job_id': job_id, 'status': 'needs_diagnosis',
                     'reason': 'Three consecutive FAILED/OUT_OF_MEMORY predecessors; auto retry stopped.',
                     'updated_utc': dt.datetime.now(dt.timezone.utc).isoformat()}
            (output/'queue_error.json').write_text(json.dumps(error, indent=2) + '\n')
            print(json.dumps(error), flush=True)
            return 0
        # Submit while this allocation is active, so a TIMEOUT/NODE_FAIL has a successor.
        try:
            successor = submit(model, profile, dependency=job_id)
        except subprocess.SubprocessError as error:
            successor = None
            (output/'queue_error.json').write_text(json.dumps({
                'job_id': job_id, 'reason': f'Could not reserve continuation: {error}',
                'updated_utc': dt.datetime.now(dt.timezone.utc).isoformat()}, indent=2) + '\n')
            print(f'Continuation submission failed; using current allocation: {error}', flush=True)
        environment = dict(os.environ, JIT_ALLOCATION_LOCK_HELD='1')
        if profile in ('single', 'standard8', 'long8'):
            command = ['torchrun', '--standalone', f'--nproc_per_node={expected_world}',
                       'scripts/run.py', '--mode', 'train', '--config', str(config)]
            if profile in ('standard8', 'long8'):
                command.append('--allow-expand-to-eight')
        else:
            hosts = subprocess.check_output(['scontrol', 'show', 'hostnames',
                os.environ['SLURM_JOB_NODELIST']], text=True, timeout=30).splitlines()
            assert len(hosts) == 2, hosts
            environment['JIT_MASTER_ADDR'] = hosts[0]
            environment['JIT_MASTER_PORT'] = str(20000 + job_id % 20000)
            command = ['srun', '--nodes=2', '--ntasks=2', '--ntasks-per-node=1',
                       '--kill-on-bad-exit=1', '--export=ALL',
                       'bash', 'scripts/torchrun_node.sh', str(config)]
        result = subprocess.run(command, cwd=ROOT, env=environment)
        if result.returncode == 0 and read(output/'summary.json', {}).get('status') == 'complete':
            # Only our own, dependency-blocked continuation is cancelled.
            if successor is not None:
                subprocess.run(['scancel', str(successor)], check=True, timeout=30)
            return 0
        return result.returncode or 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['plan', 'submit', 'worker'])
    parser.add_argument('model', choices=['b16', 'l16'])
    parser.add_argument('profile', choices=PROFILES)
    args = parser.parse_args()
    if args.action == 'plan':
        command = sbatch_command(args.model, args.profile)
        command.insert(1, '--test-only')
        print(' '.join(command), flush=True)
        sys.exit(subprocess.run(command, cwd=ROOT).returncode)
    elif args.action == 'submit':
        submit(args.model, args.profile)
    else:
        sys.exit(worker(args.model, args.profile))
