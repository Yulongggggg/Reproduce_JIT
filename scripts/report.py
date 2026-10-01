"""Build status and comparison reports from actual persisted evidence."""
import csv
import datetime
import json
from pathlib import Path
import subprocess
from resource_queue import queue_ids
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
    metadata = {m: read(p/'run_metadata.json', {}) for m,p in runs.items()}
    worlds = {m: metadata[m].get('world_size', 4) for m in MODELS}
    transitions = {m: read(p/'gpu_transition.json') for m,p in runs.items()}
    plan = read(REPORTS/'experiment_plan.json', {})
    progress = {m: read(p/'progress.json', {'completed_epochs':0,'status':'not_started'})
                for m,p in runs.items()}
    summaries = {m: read(p/'summary.json') for m,p in runs.items()}
    configs = {m: read(ROOT/f'configs/{m}_8gpu_standard_200ep.json') if worlds[m] == 8
                  else read(ROOT/f'configs/{m}_4gpu_200ep.json') for m in MODELS}
    comparisons = {m: comparison(summaries[m], configs[m]) for m in MODELS}
    jobs = read(REPORTS/'jobs_4gpu.json', {})
    data = read(ROOT/'data/imagenet/manifest.json')
    smokes = {m:read(REPORTS/f'smoke_{m}.json') for m in MODELS}
    smoke = smokes.get('b16') or read(REPORTS/'smoke.json')
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    try:
        ids = ','.join(map(str, queue_ids(jobs, REPORTS/'jobs_flexible.json')))
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
            f'## 当前进度\n\n**已完成训练与最终评估：{len(complete)}/{len(MODELS)}。** 目录沿用原四卡路径；实际训练卡数以运行元数据为准。训练轮数取自逐轮日志，每轮保存检查点，异常退出后从检查点恢复：']
    waiting = ' PENDING ' in queue and ' RUNNING ' not in queue and len(complete) < len(MODELS)
    if waiting:
        text.append('**当前候选任务在排队，没有正在运行的 JiT 训练作业。** 续跑从检查点恢复，尚未保存的日志轮数需要重跑。')
    if plan.get('long8_enabled'):
        text.append('用户最新安排：为节约 SU，JiT 训练只使用 long QoS。B/16、L/16 各申请单节点 8 张 H100/H200，单次 48 小时至 7 天；旧四卡 priority 和八卡 standard 候选已撤销且不再自动恢复。所有候选共享同模型文件锁，从现有检查点续跑。其他项目不在本次调整范围。')
    elif plan.get('enabled_profiles') == ['standard8']:
        text.append('用户最新安排：JiT 训练只使用 standard QoS。B/16、L/16 各申请单节点 8 张 H100/H200，单次 24–48 小时；旧四卡 priority 和八卡 long 候选已撤销且不再自动恢复。所有候选共享同模型文件锁，从现有检查点续跑。其他项目不在本次调整范围。')
    elif plan.get('standard8_enabled'):
        text.append('用户最新安排：保留已有四卡 priority 队列，增加 B/16、L/16 各一个单节点八卡 standard 备选，申请 24–48 小时。long 方案仅做过预检查，未提交。所有候选共享同模型文件锁；八卡实际接手后，旧四卡候选跳过该模型，不重复训练。')
    if plan.get('long8_enabled') or plan.get('standard8_enabled'):
        text.append('八卡 B/16 每卡 batch=128、累积 1 次；L/16 每卡 batch=64、累积 2 次，均保持有效 batch=1024。四卡转八卡保留模型、优化器、三组 EMA、epoch 和 global step；保留旧 rank RNG，并为新增 rank 设置独立种子。数据分片、随机轨迹和归约顺序会变化，不声称逐位一致，也不把它标为从头八卡训练。')
        text.append('扩展计划：先完成 200 epoch 的固定 CFG 和完整论文流程 FID-50K，再判断是否延长到 600 epoch。质量阈值尚未最终确定，目前没有提交或自动启动 600 epoch；200 epoch 完成后定时器继续汇报，等待扩展决定。')
    for m in MODELS:
        text.append(f'JiT-{m[0].upper()}/{m[1:]} 最近一次训练元数据记录为 {worlds[m]} 卡。'
                    if metadata[m] else f'JiT-{m[0].upper()}/{m[1:]} 尚无实际训练卡数记录。')
        if transitions[m]:
            transition = transitions[m]
            text.append(f'JiT-{m[0].upper()}/{m[1:]} 已在第 {transition["completed_epochs"]} 轮检查点后从 '
                        f'{transition["from_world_size"]} 卡切换至 {transition["to_world_size"]} 卡；'
                        f'优化器累计步数 {transition["global_step"]}，原检查点保留为 `{transition["source_checkpoint"]}`。')
    network_issue = read(REPORTS/'multinode_network_issue.json')
    network_probes = {m: read(runs[m]/'network_probe.json') for m in MODELS}
    if network_issue:
        failures = '；'.join(f'{job["model"]} 作业 {job["job_id"]}（{job["state"]}）'
                             for job in network_issue['failed_jobs'])
        text.append(f'**跨节点启动故障：{failures}。** 两个任务在美东 2026-09-29 14:37 先后获分配，'
                    '各用 4 张 H200（2 节点各 2 卡），均在初始 NCCL barrier 报 '
                    '`ibv_modify_qp: Invalid argument errno 22`，尚未进入训练，未推进检查点。'
                    '已仅为双节点入口配置 `NCCL_IB_DISABLE=1`、`NCCL_NET=Socket`，'
                    '双节点入口保留真实四卡 all-reduce 检查；当前八卡单节点任务不使用此入口。'
                    '故障证据与配置范围见 [记录](multinode_network_issue.json)。')
        for m, probe in network_probes.items():
            name = f'JiT-{m[0].upper()}/{m[1:]}'
            if probe and probe.get('status') == 'passed':
                text.append(f'{name} 通信检查：作业 {probe["slurm_job_id"]} 于 '
                            f'{probe["updated_utc"]} 通过四卡 collective 检查；这不等于完成训练或评估。')
            else:
                text.append(f'{name} TCP 替代方案尚无真实 GPU 验证结果，不能宣称故障已解决。')
    schedule = read(REPORTS/'report_schedule.json')
    if schedule:
        text.append(f'自动汇报：每 6 小时在本对话和 GitHub 更新（美东 {schedule["calendar"]}），状态 `{schedule["status"]}`；下一次计划时间：{schedule.get("next_report_local") or "任务已完成"}。定时器所在登录主机需保持运行。')
    text.extend([f'- JiT-{m[0].upper()}/{m[1:]}：训练历史日志最高完成 {progress[m]["completed_epochs"]}/200 epochs；可恢复检查点为第 {progress[m]["checkpoint_completed_epochs"]} 轮。' for m in MODELS])
    running_ids = {line.split()[0] for line in queue.splitlines() if ' RUNNING ' in line}
    for m in MODELS:
        queue_error = read(runs[m]/'queue_error.json')
        if queue_error:
            text.append(f'JiT-{m[0].upper()}/{m[1:]} 调度异常记录：{queue_error}')
        recent = [r for r in rows if r['model'] == m][-5:]
        if recent:
            seconds = sum(r['seconds'] for r in recent) / len(recent)
            model_running = str(metadata[m].get('slurm_job_id')) in running_ids
            restart_epoch = progress[m]['completed_epochs'] if model_running else progress[m]['checkpoint_completed_epochs']
            left = max(0, 200 - restart_epoch) * seconds / 3600
            text.append(f'JiT-{m[0].upper()}/{m[1:]} 最近 {len(recent)} 个已完成 epoch 平均 {seconds/60:.1f} 分钟；'
                        f'仅在相同吞吐下，剩余训练约 {left:.1f} 小时，不含评估、重试和排队。'
                        '改变卡数或节点通信方式后的速度需实测，不能直接沿用此前耗时。')
        monitors = [read(p) for p in (runs[m]/'evaluations').glob('monitor-*.json')]
        if monitors:
            metric = max(monitors, key=lambda r:r['completed_epochs'])
            text.append(f'JiT-{m[0].upper()}/{m[1:]} 第 {metric["completed_epochs"]} epoch 中途评估：FID-{metric["num_images"]}={metric["frechet_inception_distance"]:.4f}，CFG={metric["cfg"]}，EMA={metric["ema"]}。这是中途 8K 指标，不能作为最终 FID-50K 结果。')
        selection = read(runs[m]/'evaluation_selection.json')
        if selection:
            text.append(f'JiT-{m[0].upper()}/{m[1:]} 第 {selection["completed_epochs"]} epoch 的 EMA/CFG 搜索已完成 {selection["completed_candidates"]}/{selection["total_candidates"]} 组。')
    retired = read(REPORTS/'jobs_8gpu_retired.json')
    if retired and plan.get('long8_enabled'):
        text.append('此前取消的队列维持取消；新授权的八卡 long 候选登记在 jobs_flexible.json，最新 QoS 切换和取消记录见 [qos_switch_long.json](qos_switch_long.json)。')
    elif retired and plan.get('enabled_profiles') == ['standard8']:
        text.append('此前取消的队列维持取消；新八卡 standard 候选登记在 jobs_flexible.json，最新切换和取消记录见 [qos_switch_standard.json](qos_switch_standard.json)。')
    elif retired and plan.get('standard8_enabled'):
        text.append('此前取消的八卡旧队列维持取消；新授权的八卡 standard 候选单独登记在 jobs_flexible.json。')
    elif retired:
        text.append('按用户明确的要求，每个训练任务总计四卡；B/16、L/16 可各用四卡并行。遗留单任务八卡待运行队列已取消；取消记录见 [jobs_8gpu_retired.json](jobs_8gpu_retired.json)。')
    else:
        text.append('8 卡实验使用独立目录 `runs/b16_200ep`、`runs/l16_200ep`；其结果不会与本页的 4 卡实验合并。')
    if plan.get('long8_enabled'):
        text.append('资源调度：唯一启用的候选类型为 long8，B/16 与 L/16 可独立运行，每任务 world_size=8、有效 batch=1024。自动续跑同样使用 long QoS；同模型文件锁防止并发写检查点，旧 profile 的提交和启动入口已禁用。')
    elif plan.get('enabled_profiles') == ['standard8']:
        text.append('资源调度：唯一启用的候选类型为 standard8，B/16 与 L/16 可独立运行，每任务 world_size=8、有效 batch=1024。自动续跑同样使用 standard QoS；同模型文件锁防止并发写检查点，priority 和 long 的提交与启动入口已禁用。')
    elif read(REPORTS/'jobs_flexible.json'):
        text.append('资源调度：B/16 与 L/16 可独立运行；候选包括单节点 4 卡和双节点各 2 卡，均保持 world_size=4、有效 batch=1024。每模型使用文件锁防止并发写同一检查点。B/16 独立候选允许 2–12 小时，L/16 允许 3–12 小时；原串行续跑链默认 3–12 小时，当前 B/16 入口也已放宽至最短 2 小时。失败/超时自动提交的后继从检查点续跑。候选不改变模型、优化器、CFG 或评估协议。')
    resource_snapshot = read(REPORTS/'resource_snapshot.json')
    if resource_snapshot:
        text.append(f'资源检查快照（{resource_snapshot["checked_utc"]}）：'
                    f'{resource_snapshot["gpu_nodes"]} 台 GPU 节点共 {resource_snapshot["total_gpus"]} 卡，'
                    f'其中 {resource_snapshot["allocated_gpus"]} 卡已被 Slurm 分配。'
                    '这是分配计数，不是 GPU 利用率；MIXED 节点可能仅 CPU 有空闲。'
                    '详细资源和候选状态见 [resource_snapshot.json](resource_snapshot.json)。')
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
             '当前按用户要求优先训练 JiT-B/16、JiT-L/16，各 200 epochs；B/16 从已有检查点续跑，L/16 独立排队。AdamW、实际 LR=2e-4、全局有效 batch=1024、5 epoch warmup、constant LR。4 张卡时 B/16 每卡 batch 128、累积 2 次；L/16 每卡 batch 64、累积 4 次。',
             '固定 CFG 和分辨率：B/16 256² CFG 2.9（官方；另测 issue #56 的 CFG 3.6、EMA 0.9996）；L/16 256² CFG 2.4。CFG interval 均为 [0.1,1.0]，ODE solver 为 50-step Heun。',
             '固定 CFG 结果沿用作者 README：在该 CFG 下用 8K 选择 EMA，再生成 50K。另按论文 Appendix A 搜索 CFG=1.0–4.0（步长 0.1）与 EMA={0.9996,0.9998,0.9999} 共 93 组，每组 8K；选出最低 FID 对应参数后生成 50K 并计算 FID。固定 CFG 的结果和论文搜索结果分别列出。',
             '训练超参、批量和累积步数见 [`configs/`](../configs/)。B/L 的 dropout 为 0，P_mean=-0.8、P_std=0.8、t_eps=0.05、label drop=0.1、noise scale=1。4 卡等效设置和数值复现的边界见 [说明](four_gpu_equivalence.md)。',
             '\n## 同 epoch、论文评估流程的 FID-50K 对照',
             f'官方基准来自 [论文 Table 6]({PAPER_URL}#S5.T6)，均为 200 epochs、256²、FID-50K。本文 40/80/120/160 epoch 的 8K 监测值不参与该表；论文未提供这些轮数的可直接对照值。',
             '未找到 B/16、L/16 可直接对照的官方 100 epoch FID-50K；100 epoch checkpoint 保留，但正式对照选择论文明确公布的 200 epoch。没有真实 50K 结果时，不判断是否已接近官方。',
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
        comparison_rows.append({'gpu_count': worlds[m], 'run': runs[m].name, **row})
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
             '训练保存 checkpoint、三组 EMA、优化器和各 rank RNG。此前每 5 epochs 保存，增加短时段候选后四卡训练每 epoch 保存；B/16、L/16 可独立并行。',
             '200 epoch 是目标训练预算；FID 使用 1000 类均衡的 50K 样本。8K 用于训练监测和 EMA/CFG 选择，不代替正式结果。',
             '2026-09-27 加入完整论文评估流程；训练脚本在作业启动时加载，排队中的自动续跑任务会读取更新后的版本，在 200 epoch 触发最终评估。',
             '每个 50K 阶段完成后自动生成此报告并尝试提交、推送到 GitHub。单组评估结果可断点复用；推送失败记录在运行目录的 `report_publish.json`，不会中断训练。']
    (REPORTS/'REPORT.md').write_text(render_markdown(text))
    (REPORTS/'status.json').write_text(json.dumps({'updated_utc':stamp,'model_progress':progress,
        'gpu_count':8 if plan.get('enabled_profiles') in (['long8'], ['standard8']) else (None if plan.get('standard8_enabled') else 4),
        'gpu_counts_per_candidate':[8] if plan.get('enabled_profiles') in (['long8'], ['standard8']) else ([4,8] if plan.get('standard8_enabled') else [4]),
        'last_training_gpu_count_by_model':worlds,'gpu_transitions':transitions,
        'experiment_plan':plan,'models_may_run_concurrently':True,
        'network_issue':network_issue,'network_probes':network_probes,
        'completed_models':complete,'data_ready':bool(data),'smoke':bool(smoke),'complete':len(complete)==len(MODELS),'jobs':jobs},indent=2)+'\n')


if __name__=='__main__':
    main()
