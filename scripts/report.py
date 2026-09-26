"""Build status and comparison reports from actual persisted evidence."""
import csv
import datetime
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports'
MODELS = ['b16', 'l16']
PAPER_FID = {'b16':4.37, 'l16':2.79, 'h16':2.29, 'b32':4.64, 'l32':3.06, 'h32':2.51}


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def main():
    REPORTS.mkdir(exist_ok=True)
    runs = {m: ROOT/f'runs/{m}_4gpu_200ep' for m in MODELS}
    progress = {m: read(p/'progress.json', {'completed_epochs':0,'status':'not_started'})
                for m,p in runs.items()}
    summaries = {m: read(p/'summary.json') for m,p in runs.items()}
    jobs = read(REPORTS/'jobs_4gpu.json', {})
    data = read(ROOT/'data/imagenet/manifest.json')
    smokes = {m:read(REPORTS/f'smoke_{m}.json') for m in MODELS}
    smoke = smokes.get('b16') or read(REPORTS/'smoke.json')
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    try:
        ids = ','.join(str(v) for v in jobs.get('job_ids', []))
        queue = subprocess.check_output(['squeue','-h','-j',ids,'-o','%i %j %T %R'],text=True,timeout=20).strip() if ids else '尚未提交'
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        queue = '当前无法查询调度器；请查阅 jobs_4gpu.json'
    complete = [m for m in MODELS if summaries[m]]
    rows = []
    for model, run in runs.items():
        if (run/'train.jsonl').exists():
            rows.extend({**json.loads(line), 'model':model} for line in (run/'train.jsonl').read_text().splitlines() if line)
    rows = list({(r['model'],r['epoch']):r for r in rows}.values())
    rows.sort(key=lambda r:(MODELS.index(r['model']),r['epoch']))
    if rows:
        with (REPORTS/'training.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n'); writer.writeheader(); writer.writerows(rows)
    text = [f'# JiT 官方配置复现实验报告\n\n更新时间：{stamp}',
            f'## 当前进度\n\n**已完成训练与最终评估：{len(complete)}/{len(MODELS)}。** 当前展示独立的 4 卡实验，训练进度来自已保存的检查点：']
    text.extend([f'- JiT-{m[0].upper()}/{m[1:]}：{progress[m]["completed_epochs"]}/200 epochs' for m in MODELS])
    for m in MODELS:
        recent = [r for r in rows if r['model'] == m][-5:]
        if recent:
            seconds = sum(r['seconds'] for r in recent) / len(recent)
            left = max(0, 200 - progress[m]['completed_epochs']) * seconds / 3600
            text.append(f'JiT-{m[0].upper()}/{m[1:]} 最近 {len(recent)} 个 epoch 平均 {seconds/60:.1f} 分钟；剩余训练约 {left:.1f} 小时，不含评估、重试和排队。')
        monitors = [read(p) for p in (runs[m]/'evaluations').glob('monitor-*.json')]
        if monitors:
            metric = max(monitors, key=lambda r:r['completed_epochs'])
            text.append(f'JiT-{m[0].upper()}/{m[1:]} 第 {metric["completed_epochs"]} epoch 中途评估：FID-{metric["num_images"]}={metric["frechet_inception_distance"]:.4f}，CFG={metric["cfg"]}，EMA={metric["ema"]}。这是中途 8K 指标，不能作为最终 FID-50K 结果。')
    text.append('8 卡实验仍使用独立目录 `runs/b16_200ep`、`runs/l16_200ep`；其结果不会与本页的 4 卡实验合并。')
    text.append('GPU smoke：'+(f'{smoke["status"]}，{smoke.get("gpu")}' if smoke else '未完成')+'。')
    if not data:
        parts=list((ROOT/'data/imagenet/.download_parts').glob('[0-9]*'))
        text.append(f'ImageNet 准备中：下载分块 {len(parts)}/551；完整校验和解压尚未完成。')
    else:
        text.append('ImageNet 完整数据已校验、解压：1,281,167 images，1000 classes。')
    text.append(f'\nSlurm 队列：\n```text\n{queue or "训练任务已离开队列，详见完成状态。"}\n```')
    if jobs.get('pipeline_note'):
        text.append('流水线记录：'+jobs['pipeline_note'])
    text += ['## 预注册设置',
             '当前按用户要求先训练 JiT-B/16、JiT-L/16；每个模型从头训练 200 epochs，AdamW、实际 LR=2e-4、全局有效 batch=1024、5 epoch warmup、constant LR。4 张卡时 B/16 每卡 batch 128、累积 2 次；L/16 每卡 batch 64、累积 4 次。',
             '固定 CFG 和分辨率：B/16 256² CFG 2.9（官方；另测 issue #56 的 CFG 3.6、EMA 0.9996）；L/16 256² CFG 2.4。CFG interval 均为 [0.1,1.0]，ODE solver 为 50-step Heun。',
             'CFG 取自作者 README 按模型和分辨率给出的训练／评估示例。每个 CFG 固定，不跨值搜索；每个模型只在该 CFG 下用 8K 选择 EMA，再以相同 CFG 和 EMA 计算 FID-50K。',
             '训练超参、批量和累积步数见 [`configs/`](../configs/)。B/L 的 dropout 为 0，P_mean=-0.8、P_std=0.8、t_eps=0.05、label drop=0.1、noise scale=1。4 卡等效设置和数值复现的边界见 [说明](four_gpu_equivalence.md)。',
             '\n## FID-50K 对照',
             '| 模型 | 分辨率 | 固定 CFG | 200-epoch 论文 FID | 本次 EMA | 本次 FID-50K | 差值 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for m in MODELS:
        cfg=read(ROOT/f'configs/{m}_200ep.json')
        summary=summaries[m]
        r=summary['final_50k_official_cfg'][0] if summary else None
        fields=[f'JiT-{m[0].upper()}/{m[1:]}',f'{cfg["img_size"]}²',f'{cfg["cfg"]:.1f}',f'{PAPER_FID[m]:.2f}',
                str(r['ema']) if r else '—',f'{r["frechet_inception_distance"]:.4f}' if r else '—',
                f'{r["frechet_inception_distance"]-PAPER_FID[m]:+.4f}' if r else '—']
        text.append('| '+' | '.join(fields)+' |')
        if summary and summary.get('issue56_diagnostic_50k'):
            issue=summary['issue56_diagnostic_50k']
            text.append(f'\nB/16 issue #56 对照：EMA={issue["ema"]}、CFG={issue["cfg"]}、FID-50K={issue["frechet_inception_distance"]:.4f}。')
            text.append('')
        if summary:
            (REPORTS/f'{m}_results.json').write_text(json.dumps(summary,indent=2)+'\n')
    if complete:
        (REPORTS/'results.json').write_text(json.dumps({m:summaries[m] for m in complete},indent=2)+'\n')
    if len(complete)==len(MODELS):
        from PIL import Image, ImageDraw
        tile=320; canvas=Image.new('RGB',(tile*3,tile*2),'white'); draw=ImageDraw.Draw(canvas)
        for i,m in enumerate(MODELS):
            cfg=read(ROOT/f'configs/{m}_200ep.json'); r=summaries[m]['final_50k_official_cfg'][0]
            key=f'final-ep200-ema{r["ema"]}-cfg{cfg["cfg"]:.1f}-n50000'
            paths=sorted((runs[m]/'samples'/key).glob('*.png'))[:16]
            x0=(i%3)*tile; y0=(i//3)*tile
            draw.text((x0+4,y0+4),f'JiT-{m[0].upper()}/{m[1:]} CFG {cfg["cfg"]:.1f} FID {r["frechet_inception_distance"]:.2f}',fill='black')
            for j,path in enumerate(paths):
                with Image.open(path) as im:
                    canvas.paste(im.convert('RGB').resize((76,76),Image.Resampling.LANCZOS),(x0+4+(j%4)*78,y0+30+(j//4)*72))
        canvas.save(REPORTS/'samples.png')
        text.append('\n按类别均匀抽取的生成样例：\n\n![B/16 与 L/16 生成结果](samples.png)')
    elif not any(progress[m]['completed_epochs'] for m in MODELS):
        text.append('\n**训练尚未开始；目前没有本次实验的 FID。**')
    if rows:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig,axes=plt.subplots(1,2,figsize=(10,4))
            for ax,m in zip(axes.flat,MODELS):
                rs=[r for r in rows if r['model']==m]
                if rs: ax.plot([r['epoch'] for r in rs],[r['loss'] for r in rs])
                ax.set(title=f'JiT-{m[0].upper()}/{m[1:]}',xlabel='Epoch',ylabel='Velocity MSE'); ax.grid(alpha=.2)
            fig.tight_layout(); fig.savefig(REPORTS/'loss.png',dpi=150); plt.close(fig)
            text.append('\n![训练曲线](loss.png)')
        except ImportError:
            pass
    text += ['\n## 复现证据与边界',
             '上游 JiT 与定制 torch-fidelity 使用锁定 submodule；原模型、denoiser、loss 和 attention 保持上游实现。环境锁、GPU smoke 和数据验证记录见本目录。',
             '训练保存 checkpoint、三组 EMA、优化器和各 rank RNG，每 5 epochs 保存一次；本轮执行顺序为 B/16、L/16。',
             '200 epoch 是目标训练预算；FID 使用 1000 类均衡的 50K 样本。8K 只用于选 EMA，不代替正式结果。']
    (REPORTS/'REPORT.md').write_text('\n\n'.join(text)+'\n')
    (REPORTS/'status.json').write_text(json.dumps({'updated_utc':stamp,'model_progress':progress,
        'gpu_count':4,'completed_models':complete,'data_ready':bool(data),'smoke':bool(smoke),'complete':len(complete)==len(MODELS),'jobs':jobs},indent=2)+'\n')


if __name__=='__main__':
    main()
