"""Produce a Chinese status/results report from actual persisted evidence only."""
import csv
import datetime
import json
import math
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports'
RUN = ROOT / 'runs/b16_200ep'


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def main():
    REPORTS.mkdir(exist_ok=True)
    progress = read(RUN / 'progress.json', {'completed_epochs': 0, 'status': 'not_started'})
    summary = read(RUN / 'summary.json')
    jobs = read(REPORTS / 'jobs.json', {})
    data = read(ROOT / 'data/imagenet/manifest.json')
    smoke = read(REPORTS / 'smoke.json')
    try:
        ids = ','.join(str(v) for k,v in jobs.items() if k.endswith('_job_id'))
        queue = subprocess.check_output(['squeue','-h','-j',ids,'-o','%i %j %T %R'],text=True).strip() if ids else '尚未提交'
    except (FileNotFoundError, subprocess.CalledProcessError):
        queue = '当前无法查询调度器；请查阅 jobs.json'
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    text = [f'# JiT-B/16 ImageNet-256 复现报告\n\n更新时间：{stamp}\n',
            f'## 实际状态\n\n已保存训练进度：**{progress["completed_epochs"]}/200 epochs**。',
            f'数据：{"完整训练集已校验并解压" if data else "尚未就绪；下载/解压中，详见本地 logs/data-*.log"}。',
            f'GPU 冒烟测试：{smoke["status"] if smoke else "尚未完成"}。',
            f'\n```text\n{queue or "作业已离开队列；终态见下方调度器记录"}\n```\n']
    if not data:
        parts = list((ROOT/'data/imagenet/.download_parts').glob('[0-9]*'))
        text.append(f'下载进度快照：已完成 {len(parts)}/551 个分块；完整 MD5 校验和解压尚未完成。')
    if jobs.get('pipeline_note'):
        text.append('流水线记录：' + jobs['pipeline_note'])
    try:
        if ids:
            accounting = subprocess.check_output(['sacct','-X','-j',ids,'--noheader',
                '--format=JobID,JobName,State,ExitCode,Elapsed,NodeList','-P'],text=True).strip()
            text.append(f'```text\n{accounting}\n```\n')
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    text += ['## 目标和设置\n',
             '目标是从头训练 JiT-B/16 至 200 epochs，比较论文的 FID-50K **4.37**。',
             '论文原始实现为 JAX/TPU；本工程使用作者公开的 PyTorch/GPU 实现。',
             '完整参数见 [配置](../configs/b16_200ep.json)，来源、成功复现反馈和差异见 [调查报告](issue_review.md)。',
             '8 张 H100/H200，单卡 batch 128，全局 batch 1024；AdamW β=(0.9,0.95)，weight decay=0，实际 LR=2e-4，5 epoch warmup 后恒定。',
             '模型直接预测 x，以 v 空间 MSE 训练；bf16 autocast；保留上游 FP32 attention score、torch.compile 和 t_eps=0.05。',
             'EMA 同时跟踪 0.9996/0.9998/0.9999；CFG 1.0–4.0，步长 0.1，在 8K 样本上搜索；选择后用均衡覆盖 1000 类的 50K 样本确认。',
             '另评估 issue #56 的 EMA=0.9996/CFG=3.6 和 README 默认 EMA=0.9999/CFG=2.9，均用 50K 样本。',
             '预注册随机种子 0；50 步 Heun（末步 Euler），CFG interval [0.1,1.0]；作者定制 torch-fidelity 和 jit_in256_stats.npz。\n',
             '## 测量结果\n']
    rows = []
    if (RUN / 'train.jsonl').exists():
        rows = [json.loads(line) for line in (RUN/'train.jsonl').read_text().splitlines() if line]
        # A resumed epoch may appear twice in logs; retain its most recent completed observation.
        rows = list({r['epoch']: r for r in rows}.values())
        rows.sort(key=lambda r:r['epoch'])
        with (REPORTS/'training.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        text.append(f'日志最新完成 epoch：{rows[-1]["epoch"]}；loss={rows[-1]["loss"]:.6f}。checkpoint 进度可能落后于日志。')
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(7,4))
            ax.plot([r['epoch'] for r in rows], [r['loss'] for r in rows])
            ax.set(xlabel='Epoch', ylabel='Mean velocity MSE', title='JiT-B/16 training loss')
            ax.grid(alpha=.25)
            fig.tight_layout()
            fig.savefig(REPORTS/'loss.png', dpi=160)
            plt.close(fig)
            text.append('![训练曲线](loss.png)')
        except ImportError:
            pass
    if summary:
        best = summary['selected_on_8k']
        text.append(f'8K 搜索选择：EMA={best["ema"]}，CFG={best["cfg"]}。8K FID 仅用于选参，不与论文 50K 数值直接比较。\n')
        text.append('| EMA | CFG | 样本数 | FID | IS | 与 4.37 的差值 |\n|---|---|---|---|---|---|')
        for r in summary['final_50k']:
            fid = r['frechet_inception_distance']
            text.append(f'| {r["ema"]} | {r["cfg"]} | {r["num_images"]} | {fid:.4f} | {r["inception_score_mean"]:.4f} | {fid-4.37:+.4f} |')
        shutil.copy2(RUN/'summary.json', REPORTS/'results.json')
        selected = f'final-ep200-ema{best["ema"]}-cfg{best["cfg"]:.1f}-n50000'
        samples = sorted((RUN/'samples'/selected).glob('*.png'))[:64]
        if samples:
            from PIL import Image
            canvas = Image.new('RGB', (8*128, math.ceil(len(samples)/8)*128))
            for i, path in enumerate(samples):
                with Image.open(path) as picture:
                    canvas.paste(picture.resize((128,128), Image.Resampling.LANCZOS), ((i%8)*128,(i//8)*128))
            canvas.save(REPORTS/'samples.png')
            text.append('固定类别间隔抽取的生成样本（未按观感筛选；缩小显示）：\n\n![生成样本](samples.png)')
    else:
        text.append('**尚无完成 200 epochs 的 FID/IS 结果；不宣称复现成功。** 数据下载、GPU smoke、排队和训练开始均不代表训练完成。')
    evaluations = [read(p) for p in sorted((RUN/'evaluations').glob('*.json'))]
    if evaluations:
        (REPORTS/'evaluations.json').write_text(json.dumps(evaluations,indent=2)+'\n')
    if (RUN/'run_metadata.json').exists():
        shutil.copy2(RUN/'run_metadata.json', REPORTS/'run_metadata.json')
    text += ['\n## 证据与限制\n',
             '数据、checkpoint、TensorBoard 原始事件及完整日志保留在本地，不提交大文件到 GitHub。',
             'checkpoint 使用临时文件原子替换，并保留 100/200 epoch 快照、优化器和各 rank RNG；每 5 epochs 保存一次，中断可能重做至多 5 epochs。',
             '训练采样器与官方一致：DistributedSampler + drop_last，每个 epoch 1251 个 optimizer steps，200 epochs 共 250200 steps。',
             '数据顺序、额外 EMA、独立 DataLoader RNG、监测评估恢复 RNG、软件次版本及硬件差异都可能影响精确数值，因此以真实 FID 和设置差异报告，不保证等于 4.37。',
             '仅单一训练 seed；生成的主评估选参依照论文 8K 搜索规则，不能用多个 50K 结果反向选择最小值作为主结果。']
    (REPORTS/'REPORT.md').write_text('\n\n'.join(text)+'\n')
    (REPORTS/'status.json').write_text(json.dumps({'updated_utc':stamp,'checkpoint_progress':progress,
        'data_ready':bool(data),'smoke':bool(smoke),'complete':bool(summary),'jobs':jobs},indent=2)+'\n')


if __name__ == '__main__':
    main()
