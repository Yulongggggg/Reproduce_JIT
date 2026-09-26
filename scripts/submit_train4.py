"""Queue a resumable four-GPU B/16 then L/16 run."""
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / 'reports/jobs_4gpu.json'
LINKS = 16


def save(record):
    temporary = RECORD.with_suffix('.tmp')
    temporary.write_text(json.dumps(record, indent=2) + '\n')
    temporary.replace(RECORD)


def main():
    if RECORD.exists():
        raise SystemExit(f'{RECORD} already exists; inspect existing jobs before submitting again')
    record = {
        'purpose': 'JiT-B/16 then JiT-L/16, 200 epochs each, four GPUs',
        'global_batch': 1024,
        'learning_rate': 0.0002,
        'config_files': ['configs/b16_4gpu_200ep.json', 'configs/l16_4gpu_200ep.json'],
        'script': 'scripts/train4.sbatch',
        'dependency': 'afternotok',
        'job_ids': [],
    }
    save(record)
    for _ in range(LINKS):
        cmd = ['sbatch', '--parsable']
        if record['job_ids']:
            cmd.append(f"--dependency=afternotok:{record['job_ids'][-1]}")
        cmd.append('scripts/train4.sbatch')
        job_id = int(subprocess.check_output(cmd, text=True, cwd=ROOT).strip().split(';')[0])
        record['job_ids'].append(job_id)
        save(record)
        print(job_id, flush=True)


if __name__ == '__main__':
    main()
