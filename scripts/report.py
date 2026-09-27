"""Build status and comparison reports from actual persisted evidence."""
import csv
import datetime
import json
from pathlib import Path
import subprocess
from evaluation_protocol import (comparison, validate_metric, PAPER_URL, PAPER_PROTOCOL,
                                 EMA_CANDIDATES, CFG_CANDIDATES)

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports'
MODELS = ['b16', 'l16']


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def render_markdown(blocks):
    # Blank lines between Markdown table rows break GitHub's table rendering.
    rendered = ''
    previous = ''
    for block in blocks:
        if rendered:
            rendered += '\n' if block.startswith('|') and previous.startswith('|') else '\n\n'
        rendered += block
        previous = block
    return rendered + '\n'


def main():
    REPORTS.mkdir(exist_ok=True)
    runs = {m: ROOT/f'runs/{m}_4gpu_200ep' for m in MODELS}
    progress = {m: read(p/'progress.json', {'completed_epochs':0,'status':'not_started'})
                for m,p in runs.items()}
    summaries = {m: read(p/'summary.json') for m,p in runs.items()}
    configs = {m: read(ROOT/f'configs/{m}_4gpu_200ep.json') for m in MODELS}
    comparisons = {m: comparison(summaries[m], configs[m]) for m in MODELS}
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
    complete = [m for m in MODELS if summaries[m] and summaries[m].get('status') == 'complete'
                and comparisons[m]['status'] == 'matched']
    rows = []
    for model, run in runs.items():
        if (run/'train.jsonl').exists():
            rows.extend({**json.loads(line), 'model':model} for line in (run/'train.jsonl').read_text().splitlines() if line)
    rows = list({(r['model'],r['epoch']):r for r in rows}.values())
    rows.sort(key=lambda r:(MODELS.index(r['model']),r['epoch']))
    for model in MODELS:
        saved = progress[model]
        saved['checkpoint_completed_epochs'] = saved['completed_epochs']
        saved['checkpoint_global_step'] = saved.get('global_step', 0)
        saved['checkpoint_updated_utc'] = saved.pop('updated_utc', None)
        recorded = [r for r in rows if r['model'] == model]
        if recorded and recorded[-1]['epoch'] > saved['completed_epochs']:
            saved['completed_epochs'] = recorded[-1]['epoch']
            saved['global_step'] = recorded[-1]['global_step']
    if rows:
        with (REPORTS/'training.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n'); writer.writeheader(); writer.writerows(rows)
    text = [f'# JiT 官方配置复现实验报告\n\n更新时间：{stamp}',
            f'## 当前进度\n\n**已完成训练与最终评估：{len(complete)}/{len(MODELS)}。** 当前展示独立的 4 卡实验。训练轮数取自逐轮日志；检查点每 5 轮保存，异常退出后从检查点恢复：']
    text.extend([f'- JiT-{m[0].upper()}/{m[1:]}：训练日志已完成 {progress[m]["completed_epochs"]}/200 epochs；可恢复检查点为第 {progress[m]["checkpoint_completed_epochs"]} 轮。' for m in MODELS])
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
        selection = read(runs[m]/'evaluation_selection.json')
        if selection:
            text.append(f'JiT-{m[0].upper()}/{m[1:]} 第 {selection["completed_epochs"]} epoch 的 EMA/CFG 搜索已完成 {selection["completed_candidates"]}/{selection["total_candidates"]} 组。')
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
             '固定 CFG 结果沿用作者 README：在该 CFG 下用 8K 选择 EMA，再生成 50K。另按论文 Appendix A 搜索 CFG=1.0–4.0（步长 0.1）与 EMA={0.9996,0.9998,0.9999} 共 93 组，每组 8K；选出最低 FID 对应参数后生成 50K 并计算 FID。固定 CFG 的结果和论文搜索结果分别列出。',
             '训练超参、批量和累积步数见 [`configs/`](../configs/)。B/L 的 dropout 为 0，P_mean=-0.8、P_std=0.8、t_eps=0.05、label drop=0.1、noise scale=1。4 卡等效设置和数值复现的边界见 [说明](four_gpu_equivalence.md)。',
             '\n## 同 epoch、论文评估流程的 FID-50K 对照',
             f'官方基准来自 [论文 Table 6]({PAPER_URL}#S5.T6)，均为 200 epochs、256²、FID-50K。本文 40/80/120/160 epoch 的 8K 监测值不参与该表；论文未提供这些轮数的可直接对照值。',
             '两边使用 1000 类、每类 50 张生成图片，对 ImageNet 训练集计算 FID。复现使用作者发布的 `jit_in256_stats.npz`、锁定的 torch-fidelity、50-step Heun、CFG interval [0.1,1.0]。评估 JSON 记录统计文件 SHA-256、后端 commit、CFG、EMA、epoch 和 checkpoint 标识。论文数值来自原实现；按公开 PyTorch 代码对齐评估流程，不承诺逐位一致。',
             '| 模型 | epoch（双方） | 分辨率 | 官方 FID-50K | 本次 FID-50K | 本次 CFG / EMA | 差值 |',
             '|---|---:|---:|---:|---:|---|---:|']
    comparison_rows = []
    for m in MODELS:
        row = comparisons[m]
        matched = row['status'] == 'matched'
        fields = [row['model'], str(row['completed_epochs']), f'{row["resolution"]}²',
                  f'{row["paper_fid"]:.2f}', f'{row["our_fid"]:.4f}' if matched else '待评估',
                  f'{row["cfg"]} / {row["ema"]}' if matched else '—',
                  f'{row["delta"]:+.4f}' if matched else '—']
        text.append('| '+' | '.join(fields)+' |')
        comparison_rows.append({'gpu_count': 4, 'run': runs[m].name, **row})
    text.extend(['\n### README 固定 CFG 的 FID-50K',
                 '此表保留指定 CFG 的结果。论文的最优 EMA/CFG 随训练轮数变化，因此本表不计算论文差值；完整搜索结果见上表。',
                 '| 模型 | epoch | 固定 CFG | 选中 EMA | FID-50K |',
                 '|---|---:|---:|---:|---:|'])
    warnings = []
    for m in MODELS:
        cfg, summary = configs[m], summaries[m]
        r = None
        if summary and summary.get('final_50k_official_cfg'):
            try:
                r = summary['final_50k_official_cfg'][0]
                validate_metric(r, cfg)
                if r['cfg'] != cfg['cfg']:
                    raise ValueError('Fixed CFG result uses a different CFG')
            except (ValueError, KeyError, TypeError) as error:
                warnings.append(f'{cfg["model"]} 固定 CFG 结果未纳入对照：{error}')
                r = None
        text.append(f'| {cfg["model"]} | {cfg["epochs"]} | {cfg["cfg"]} | '
                    + (f'{r["ema"]} | {r["frechet_inception_distance"]:.4f} |' if r else '— | 待评估 |'))
        if comparisons[m]['status'] == 'ineligible':
            warnings.append(f'{cfg["model"]} 论文对照未通过校验：{comparisons[m]["reason"]}')
        if summary:
            (REPORTS/f'{m}_results.json').write_text(json.dumps(summary,indent=2)+'\n')
    for m in MODELS:
        summary = summaries[m]
        if summary and summary.get('issue56_diagnostic_50k'):
            issue = summary['issue56_diagnostic_50k']
            try:
                validate_metric(issue, configs[m])
                if (issue['ema'], issue['cfg']) != (0.9996, 3.6):
                    raise ValueError('Issue #56 parameters do not match')
                text.append(f'\nB/16 issue #56 单独对照：epoch={issue["completed_epochs"]}，EMA={issue["ema"]}、CFG={issue["cfg"]}、FID-50K={issue["frechet_inception_distance"]:.4f}。')
            except (ValueError, KeyError, TypeError) as error:
                warnings.append(f'Issue #56 结果未通过校验：{error}')
        # Independent eight-GPU runs must also be published automatically if they finish.
        other = ROOT/f'runs/{m}_200ep'
        summary8 = read(other/'summary.json')
        if summary8:
            row8 = comparison(summary8, read(ROOT/f'configs/{m}_200ep.json'))
            comparison_rows.append({'gpu_count': 8, 'run': other.name, **row8})
            (REPORTS/f'{m}_8gpu_results.json').write_text(json.dumps(summary8,indent=2)+'\n')
    if len(comparison_rows) > len(MODELS):
        text.extend(['\n### 独立 8 卡实验（同样校验 200 epoch / FID-50K）',
                     '| 模型 | 官方 FID-50K | 本次 FID-50K | 差值 |', '|---|---:|---:|---:|'])
        for row in comparison_rows[len(MODELS):]:
            text.append(f'| {row["model"]} | {row["paper_fid"]:.2f} | '
                        + (f'{row["our_fid"]:.4f} | {row["delta"]:+.4f} |'
                           if row['status'] == 'matched' else f'待评估 | — |'))
    text.extend('\n'+warning for warning in warnings)
    (REPORTS/'comparison_50k.json').write_text(json.dumps(comparison_rows,indent=2)+'\n')
    with (REPORTS/'comparison_50k.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=['run','gpu_count','model','resolution',
            'completed_epochs','num_images','paper_fid','our_fid','cfg','ema','delta',
            'status','reason','paper_source'], extrasaction='ignore', lineterminator='\n')
        writer.writeheader(); writer.writerows(comparison_rows)
    (REPORTS/'evaluation_protocol.json').write_text(json.dumps({
        'policy': PAPER_PROTOCOL, 'paper_source': PAPER_URL + '#A1',
        'selection_num_images': 8000, 'final_num_images': 50000,
        'trigger_completed_epochs': 200,
        'class_num': 1000, 'final_samples_per_class': 50,
        'ema_candidates': EMA_CANDIDATES, 'cfg_candidates': CFG_CANDIDATES,
        'fixed_cfg': {configs[m]['model']: configs[m]['cfg'] for m in MODELS},
        'rollout_note': 'Runner loads at job start; queued continuation jobs read the updated evaluation code.',
        'report_publication': 'automatic after each completed 50K stage; push failures recorded in run/report_publish.json',
    },indent=2)+'\n')
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
             '200 epoch 是目标训练预算；FID 使用 1000 类均衡的 50K 样本。8K 用于训练监测和 EMA/CFG 选择，不代替正式结果。',
             '2026-09-27 加入完整论文评估流程；训练脚本在作业启动时加载，排队中的自动续跑任务会读取更新后的版本，在 200 epoch 触发最终评估。',
             '每个 50K 阶段完成后自动生成此报告并尝试提交、推送到 GitHub。单组评估结果可断点复用；推送失败记录在运行目录的 `report_publish.json`，不会中断训练。']
    (REPORTS/'REPORT.md').write_text(render_markdown(text))
    (REPORTS/'status.json').write_text(json.dumps({'updated_utc':stamp,'model_progress':progress,
        'gpu_count':4,'completed_models':complete,'data_ready':bool(data),'smoke':bool(smoke),'complete':len(complete)==len(MODELS),'jobs':jobs},indent=2)+'\n')


if __name__=='__main__':
    main()
