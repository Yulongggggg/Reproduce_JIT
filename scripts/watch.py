"""Persistently monitor this experiment, publish reports, retry infrastructure failures."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
JOBS = ROOT / 'reports/jobs.json'


def state(job):
    result = subprocess.check_output(['sacct','-X','-n','-j',str(job),
        '--format=JobIDRaw,State','-P'], text=True)
    lines = [row.split('|') for row in result.splitlines() if row.strip()]
    return next((s.split()[0].rstrip('+') for i,s,*rest in lines if i == str(job)), 'UNKNOWN')


def write_jobs(jobs):
    tmp = JOBS.with_suffix('.tmp')
    tmp.write_text(json.dumps(jobs, indent=2)+'\n')
    tmp.replace(JOBS)


def main():
    os.chdir(ROOT)
    lock = open(ROOT/'artifacts/watcher.lock','w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    last_publish, previous = 0, None
    terminal = {'COMPLETED','FAILED','CANCELLED','TIMEOUT','NODE_FAIL','OUT_OF_MEMORY','PREEMPTED','BOOT_FAIL'}
    retryable = {'TIMEOUT','NODE_FAIL','PREEMPTED','BOOT_FAIL'}
    while True:
        try:
            jobs = json.loads(JOBS.read_text())
            states = {k:state(v) for k,v in jobs.items() if k.endswith('_job_id')}
            print(time.strftime('%Y-%m-%d %H:%M:%S'), states, flush=True)
            data_ok = states.get('data_job_id') == 'COMPLETED'
            smoke_ok = states.get('smoke_job_id') == 'COMPLETED'
            dependencies_failed = any(states.get(k) in terminal - {'COMPLETED'}
                                      for k in ('data_job_id','smoke_job_id'))
            train_state = states.get('train_job_id')
            if dependencies_failed and train_state not in terminal:
                subprocess.run(['scancel', str(jobs['train_job_id'])], check=True)
                jobs['pipeline_note'] = 'Preparation or smoke failed; dependent training cancelled. Inspect logs.'
                write_jobs(jobs)
            elif data_ok and smoke_ok and train_state in retryable and jobs.get('retries', 0) < 3:
                new_job = subprocess.check_output(['sbatch','--parsable','scripts/train.sbatch'],text=True).strip().split(';')[0]
                jobs.setdefault('previous_train_jobs', []).append(jobs['train_job_id'])
                jobs['train_job_id'] = int(new_job)
                jobs['retries'] = jobs.get('retries', 0) + 1
                jobs['pipeline_note'] = f'Resuming from saved checkpoint after {train_state}.'
                write_jobs(jobs)
                train_state = 'PENDING'
            completed = (ROOT/'runs/b16_200ep/summary.json').exists()
            done = completed or dependencies_failed or train_state in terminal
            if train_state == 'COMPLETED' and not completed:
                jobs['pipeline_note'] = 'Slurm completed but final summary missing; NOT a completed reproduction.'
                write_jobs(jobs)
            if states != previous or time.time() - last_publish >= 21600 or done:
                published = subprocess.run(['bash','scripts/publish.sh']).returncode == 0
                if published:
                    last_publish, previous = time.time(), states
                else:
                    print('Report publication failed; no force push; will retry.', flush=True)
                    if done:
                        # Keep local report and an explicit sync-failure record.
                        (ROOT/'artifacts/publish_failed.txt').write_text('Final report exists locally; git push failed.\n')
            if done:
                print('Monitoring stopped at terminal state; inspect reports/REPORT.md.', flush=True)
                return
        except Exception as error:
            print(f'Watcher error: {type(error).__name__}: {error}', flush=True)
        time.sleep(60)


if __name__ == '__main__':
    main()
