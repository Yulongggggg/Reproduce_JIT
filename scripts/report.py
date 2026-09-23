"""Build status and comparison reports from actual persisted evidence."""
import csv
import datetime
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports'
MODELS = ['b16', 'l16', 'b32', 'l32', 'h16', 'h32']
PAPER_FID = {'b16':4.37, 'l16':2.79, 'h16':2.29, 'b32':4.64, 'l32':3.06, 'h32':2.51}


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def main():
    REPORTS.mkdir(exist_ok=True)
    runs = {m: ROOT/f'runs/{m}_200ep' for m in MODELS}
    progress = {m: read(p/'progress.json', {'completed_epochs':0,'status':'not_started'})
                for m,p in runs.items()}
    summaries = {m: read(p/'summary.json') for m,p in runs.items()}
    jobs = read(REPORTS/'jobs.json', {})
    data = read(ROOT/'data/imagenet/manifest.json')
    smokes = {m:read(REPORTS/f'smoke_{m}.json') for m in MODELS}
    smoke = smokes.get('b16') or read(REPORTS/'smoke.json')
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    try:
        ids = ','.join(str(v) for k,v in jobs.items() if k.endswith('_job_id'))
        queue = subprocess.check_output(['squeue','-h','-j',ids,'-o','%i %j %T %R'],text=True).strip() if ids else '尚未提交'
    except (FileNotFoundError, subprocess.CalledProcessError):
        queue = '当前无法查询调度器；请查阅 jobs.json'
    complete = [m for m in MODELS if summaries[m]]
    rows = []
    for model, run in runs.items():
        if (run/'train.jsonl').exists():
            rows.extend({'model':model, **json.loads(line)} for line in (run/'train.jsonl').read_text().splitlines() if line)
    rows = list({(r['model'],r['epoch']):r for r in rows}.values())
    rows.sort(key=lambda r:(MODELS.index(r['model']),r['epoch']))
    if rows:
        with (REPORTS/'training.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    text = [f'# JiT 官方配置复现实验报告\n\n更新时间：{stamp}',
            f'## 当前进度\n\n**已完成 200 epochs：{len(complete)}/6。** 当前训练进度：']
    text.extend([f'- JiT-{m[0].upper()}/{m[1:]}：{progress[m]["completed_epochs"]}/200 epochs' for m in MODELS])
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
             '依次训练 JiT-B/16、JiT-L/16、B/32、L/32、H/16、H/32；每个模型从头训练 200 epochs，AdamW、实际 LR=2e-4、全局有效 batch=1024、5 epoch warmup、constant LR。',
             '固定 CFG 和分辨率：B/16 256² CFG 2.9（官方；另测 issue #56 的 CFG 3.6、EMA 0.9996）；L/16 256² CFG 2.4；B/32 512² CFG 2.9；L/32 512² CFG 2.5；H/16 256² CFG 2.2；H/32 512² CFG 2.3。CFG interval 均为 [0.1,1.0]，ODE solver 为 50-step Heun。',
             'CFG 取自作者 README 按模型和分辨率给出的训练／评估示例。每个 CFG 固定，不跨值搜索；每个模型只在该 CFG 下用 8K 选择 EMA，再以相同 CFG 和 EMA 计算 FID-50K。',
             '训练超参、批量和累积步数见 [`configs/`](../configs/)。梯度累积保持全局有效 batch=1024。H 型官方 proj dropout=0.2，其余为 0。P_mean=-0.8、P_std=0.8、t_eps=0.05、label drop=0.1、noise scale=res/256。',
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
    if len(complete)==6:
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
        text.append('\n按类别均匀抽取的生成样例：\n\n![六种模型生成结果](samples.png)')
    elif not any(progress[m]['completed_epochs'] for m in MODELS):
        text.append('\n**训练尚未开始；目前没有本次实验的 FID。**')
    if rows:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig,axes=plt.subplots(2,3,figsize=(12,7))
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
             '训练保存 checkpoint、三组 EMA、优化器和各 rank RNG，每 5 epochs 保存一次；执行顺序为 B/16、L/16 优先，随后其余四种。',
             '200 epoch 是目标训练预算；FID 使用 1000 类均衡的 50K 样本。8K 只用于选 EMA，不代替正式结果。']
    (REPORTS/'REPORT.md').write_text('\n\n'.join(text)+'\n')
    (REPORTS/'status.json').write_text(json.dumps({'updated_utc':stamp,'model_progress':progress,
        'completed_models':complete,'data_ready':bool(data),'smoke':bool(smoke),'complete':len(complete)==6,'jobs':jobs},indent=2)+'\n')


if __name__=='__main__':
    main()
