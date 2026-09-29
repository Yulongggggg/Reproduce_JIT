"""Six-hour report publisher and notification to the existing, user-authorized chat.

Invoked by a user systemd timer on the login host. Never submits/cancels GPU jobs.
Private thread identifiers and delivery receipts stay in ignored artifacts/.
"""
import argparse
import datetime as dt
import fcntl
import json
from pathlib import Path
import socket
import subprocess
import time
from zoneinfo import ZoneInfo

from evaluation_protocol import comparison
from resource_queue import queue_ids

ROOT = Path(__file__).resolve().parents[1]
INTERVAL = 21600
TIMEZONE = ZoneInfo('America/New_York')
UNIT = 'jit-report-6h.timer'
PROMPT = '''【用户已授权的 JiT 每 6 小时定时汇报】
请在本对话给用户一份简洁的中文进度报告，并更新 /mnt/home/yliu5/Reproduce_JIT 的 GitHub 报告。
读取真实 Slurm 状态、训练日志、检查点和评估文件，区分正在运行、排队和故障；说明 B/16 与 L/16 各自进度、GPU 数及剩余时间。超时后可能从较早检查点重跑，不能把历史日志最大轮数当成当前正在执行的轮数。
仅用真实的、同模型/分辨率/epoch/评估协议的 FID-50K 与论文比较，给出绝对差值及百分比。官方 Table 6 的 200 epoch 基准：B/16=4.37，L/16=2.79。未找到可直接对照的官方 100 epoch 基准，不拿 100 epoch 对比 200 epoch。8K 仅是监测或选参数据，不可冒充 50K；没有正式结果时直说尚不能判断是否接近官方。
保留 B/16 CFG=2.9、L/16 CFG=2.4 的固定 CFG 结果，并分别报告论文 EMA/CFG 搜索流程的结果。必要时排查真实故障，但不重复提交运行/排队中的训练，不随意修改训练超参。
用户已授权主动寻找资源并增加候选。同时读取 reports/jobs_flexible.json，追踪单节点四卡/双节点各两卡候选；它们使用同模型互斥锁，可让 B/16、L/16 并行。发现某模型仍未完成却已无运行/待运行候选时，可通过 scripts/resource_queue.py 的幂等 submit 恢复候选，禁止绕过模型锁。固定 world_size=4、有效 batch=1024，不重复从头训练。
用户已明确是每个训练任务总计四卡，B/16、L/16 可各用四卡并行，并非整个项目只能同时用四卡。遗留单任务八卡队列已取消，不得重新提交或恢复单任务八卡任务。
本次是已安装定时器的正常触发，不要重新创建定时器，也不需要再次询问汇报授权。使用 bash scripts/publish.sh 更新 GitHub。B/16、L/16 的 200 epoch 训练和正式对照均完成后，给最终总结并确认此定时器已经停止。'''


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def next_due(now):
    """02:30, 08:30, 14:30, 20:30 Eastern, with one missed tick coalesced by systemd."""
    local = dt.datetime.fromtimestamp(now, TIMEZONE)
    for day in range(2):
        for hour in (2, 8, 14, 20):
            candidate = (local + dt.timedelta(days=day)).replace(
                hour=hour, minute=30, second=0, microsecond=0)
            if candidate.timestamp() > now:
                return candidate.isoformat()
    raise AssertionError('No next scheduled time')


def finished():
    for model in ('b16', 'l16'):
        summary = read(ROOT/f'runs/{model}_4gpu_200ep/summary.json')
        cfg = read(ROOT/f'configs/{model}_4gpu_200ep.json')
        if not summary or summary.get('status') != 'complete':
            return False
        if comparison(summary, cfg)['status'] != 'matched':
            return False
    return True


def refresh_jobs():
    path = ROOT/'reports/jobs_4gpu.json'
    jobs = read(path)
    all_ids = queue_ids(jobs, ROOT/'reports/jobs_flexible.json')
    ids = ','.join(map(str, all_ids))
    output = subprocess.check_output(['squeue', '-h', '-j', ids, '-o', '%i|%T|%R'],
                                     text=True, timeout=30)
    states = {}
    for line in output.splitlines():
        job, state, reason = line.split('|', 2)
        states[int(job)] = {'state': state, 'node_or_reason': reason}
    running = [i for i in all_ids if states.get(i, {}).get('state') == 'RUNNING']
    pending = [i for i in all_ids if states.get(i, {}).get('state') == 'PENDING']
    jobs['active_train_job_ids'] = running
    jobs['additional_job_ids'] = [i for i in all_ids if i not in jobs['job_ids']]
    jobs['active_train_job_id'] = running[0] if running else None
    jobs['next_train_job_id'] = pending[0] if pending else None
    jobs['queue_checked_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
    jobs['queue_snapshot'] = states
    accounting = subprocess.check_output(['sacct', '-X', '-n', '-P', '-j', ids,
        '--format=JobIDRaw,State'], text=True, timeout=30)
    timed_out = [int(line.split('|')[0]) for line in accounting.splitlines()
                 if len(line.split('|')) > 1 and line.split('|')[1].startswith('TIMEOUT')
                 and line.split('|')[0].isdigit()]
    if timed_out:
        jobs['last_timed_out_job_id'] = max(timed_out)
    stamp = dt.datetime.now(TIMEZONE).strftime('%Y-%m-%d %H:%M %Z')
    if running:
        status = '正在运行：' + '；'.join(f'{i} ({states[i]["node_or_reason"]})' for i in running) + '。'
    elif pending:
        status = f'目前没有四卡训练作业运行；续跑作业 {pending[0]} 正在排队（{states[pending[0]]["node_or_reason"]}）。'
    else:
        status = '四卡训练链当前没有运行或排队的作业；需结合完成状态判断是否结束或异常。'
    jobs['pipeline_note'] = f'{stamp} 自动检查：{status} 训练历史最大轮数和可恢复检查点分别列出。'
    write(path, jobs)


def command(argv, timeout):
    try:
        result = subprocess.run(argv, cwd=ROOT, text=True, capture_output=True, timeout=timeout)
        return {'returncode': result.returncode, 'stdout': result.stdout[-5000:],
                'stderr': result.stderr[-5000:]}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {'returncode': -1, 'error': str(error)}


def deliver(config, receipt, slot, run=command):
    """Retry a failed publication without sending the same chat notification twice."""
    if receipt.get('slot') != slot:
        receipt.clear()
        receipt['slot'] = slot
    if receipt.get('publication', {}).get('returncode') != 0:
        receipt['publication'] = run(['bash', 'scripts/publish.sh'], 300)
    if receipt.get('notification', {}).get('returncode') != 0:
        receipt['notification'] = run([config['codex_bin'], 'queue', '--thread',
            config['thread_id'], '--message', PROMPT], 90)
    return all(receipt.get(k, {}).get('returncode') == 0 for k in ('publication', 'notification'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--notify-test-only', action='store_true')
    parser.add_argument('--refresh-only', action='store_true')
    args = parser.parse_args()
    config = read(ROOT/'artifacts/report_schedule_config.json')
    if not config:
        raise SystemExit('Missing private artifacts/report_schedule_config.json')
    if config['host'] != socket.gethostname():
        raise SystemExit('Run this timer on its configured login host to reach the existing Codex chat')
    if args.refresh_only:
        refresh_jobs()
        return
    if args.notify_test_only:
        result = command([config['codex_bin'], 'queue', '--thread', config['thread_id'],
            '--message', '【JiT 定时汇报连通性测试】用户已授权每 6 小时汇报。本条仅验证消息能返回当前对话；继续完成本轮定时器安装，不重复启动任务、不需要另开汇报回合。'], 90)
        write(ROOT/'artifacts/report_notification_test.json', result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        raise SystemExit(0 if result['returncode'] == 0 else 1)
    lock = (ROOT/'artifacts/report_tick.lock').open('w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return
    now = time.time()
    complete = finished()
    status = {'status': 'completed' if complete else 'active', 'interval_hours': 6,
              'timezone': str(TIMEZONE), 'calendar': '02:30, 08:30, 14:30, 20:30',
              'host': config['host'], 'systemd_timer': UNIT,
              'last_attempt_utc': dt.datetime.fromtimestamp(now, dt.timezone.utc).isoformat(),
              'next_report_local': None if complete else next_due(now),
              'destinations': ['existing Codex conversation', 'GitHub reports/REPORT.md'],
              'comparison_epoch': 200, 'comparison_images': 50000,
              'official_100_epoch_reference': None,
              'requires': 'login host and user systemd service running; Codex daemon reachable'}
    try:
        refresh_jobs()
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        status['scheduler_query_error'] = str(error)
    write(ROOT/'reports/report_schedule.json', status)
    receipt_path = ROOT/'artifacts/report_delivery.json'
    receipt = read(receipt_path, {})
    success = deliver(config, receipt, int(now // INTERVAL))
    receipt['updated_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
    write(receipt_path, receipt)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)
    if not success:
        raise SystemExit(1)  # systemd retries in five minutes; successful channel is not repeated.
    if complete:
        subprocess.run(['systemctl', '--user', 'disable', '--now', UNIT], check=True, timeout=30)


if __name__ == '__main__':
    main()
