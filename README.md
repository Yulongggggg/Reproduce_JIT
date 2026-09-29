# Reproduce JiT

ImageNet-1K 上复现 JiT，当前按用户要求先跑 **B/16、L/16，各 200 epochs**。保留 README 固定 CFG 的 FID-50K，并补充论文 EMA/CFG 搜索流程的 FID-50K。两类结果分别报告；B/16 另保留 issue #56 的 CFG=3.6/EMA=0.9996 对照。

- [训练状态与同 epoch 的 FID-50K 对照](reports/REPORT.md)
- [机器可读的 FID-50K 对照表](reports/comparison_50k.csv)
- [评估协议](reports/evaluation_protocol.json)
- [复现前 issue 调查](reports/issue_review.md)
- [逐模型参数配置](configs/)

当前工作目录：`/mnt/home/yliu5/Reproduce_JIT`。

## 官方 README 的逐模型 CFG

```text
B/16, 256², CFG 2.9     issue #56 comparison: CFG 3.6
L/16, 256², CFG 2.4
B/32, 512², CFG 2.9
L/32, 512², CFG 2.5
H/16, 256², CFG 2.2
H/32, 512², CFG 2.3
```

当前所有训练任务均使用总计四卡复现 B/16、L/16，独立候选让两模型可并行。按用户最新要求，原 8 卡待运行队列已取消，不再提交。B/16 每卡 batch=128、累积 2 次，L/16 每卡 batch=64、累积 4 次，均为有效 batch=1024。其他四个模型仅保留配置，暂不进入当前队列。

## 自动 FID-50K

每个模型在 200 epoch 后执行：

1. 固定 B/16 CFG=2.9、L/16 CFG=2.4，用 8K 选 EMA，计算 FID-50K。
2. 按[论文 Appendix A](https://arxiv.org/html/2511.13720v2#A1)，在 CFG=1.0–4.0（步长 0.1）和三个 EMA（0.9996/0.9998/0.9999）间做 93 组 8K 评估，再用选出的参数计算 FID-50K。
3. 正式评估均生成 1000 类 × 每类 50 张，采用作者的 FID 统计文件、torch-fidelity、50-step Heun 和 CFG interval [0.1,1.0]；记录统计文件 hash、后端版本和 checkpoint 标识。
4. 自动更新、提交并推送报告；对照[论文 Table 6](https://arxiv.org/html/2511.13720v2#S5.T6) 的 **200 epoch** B/16=4.37、L/16=2.79。样本数、轮数或协议不匹配时不计算差值。论文未报告的中间轮数不填造基准。

93 组搜索共需生成 744K 图片，评估时间需另计；每组结果单独保存，超时后复用。8K 是监测/选参分数，正式对照全部使用 50K。固定 CFG 表不声称与论文的搜索流程相同。报告推送失败不会中断训练，错误写入各运行目录的 `report_publish.json`。

## 每 6 小时汇报

`jit-report-6h.timer` 在登录主机的用户 systemd 服务中运行，美东时间每天 **02:30、08:30、14:30、20:30** 调用 `scripts/report_tick.py`。它刷新 Slurm 状态、推送 GitHub 报告，并通过 `codex queue` 唤起当前对话汇报；不提交或取消训练作业。服务器和 Codex 服务需要保持可用。失败后每 5 分钟重试，已成功的消息投递不会在同一时段重复发送。B/16、L/16 均完成 200 epoch 和经校验的 FID-50K 后自动停止。

使用 `systemctl --user list-timers jit-report-6h.timer` 查看下次执行时间；使用 `systemctl --user disable --now jit-report-6h.timer` 停止。对话标识、投递回执保存在被 Git 忽略的 `artifacts/`；公开状态见 `reports/report_schedule.json`。

没有找到上述两个模型可直接对照的官方 100 epoch FID-50K，因此正式比较统一使用 200 epoch，不能用 100 epoch 或 FID-8K 冒充同设置对照。

## 运行与恢复

```bash
git clone --recurse-submodules https://github.com/Yulongggggg/Reproduce_JIT.git
cd Reproduce_JIT
bash scripts/install.sh
mkdir -p logs
data_job=$(sbatch --parsable scripts/data.sbatch)
smoke_job=$(sbatch --parsable scripts/smoke.sbatch)
sbatch --dependency=afterok:${data_job}:${smoke_job} scripts/train4.sbatch
```

Slurm job script 对应当前集群账号。原四卡脚本为 `scripts/train4.sbatch`，申请 4 张 H100/H200、40 CPU、256 GiB RAM，每次最多 12 小时，已有自动续跑链（见 `reports/jobs_4gpu.json`）。数据已通过完整 MD5，并解压为 1000 类、1,281,167 张训练图片。

### 可并行的资源候选

用户授权增加候选后，`scripts/resource_queue.py` 支持 `single`（一节点四卡）和 `split`（两节点各两卡）两种申请。两者总计均为四卡、40 CPU、256 GiB RAM，保持原配置、四个全局 rank 和有效 batch=1024；跨节点会改变通信性能，不声称数值逐位一致。申请 `priority` QoS；B/16 独立候选允许 2–12 小时，L/16 允许 3–12 小时，便于 Slurm backfill 提前安排较短空档。原串行续跑链默认最短 3 小时；B/16 已到 184 轮后，当前入口 101984 的最短时段也放宽到 2 小时，保留其排队资历。

每模型的 `.allocation.lock` 覆盖原入口与新入口：同时分配到资源的候选中，仅一个能训练/评估并写该模型检查点，另一个立即释放；B/16 和 L/16 使用不同锁，可以并行。新入口先提交唯一的 `afternotok` 后继，再运行训练；超时/失败后自动续跑，模型全部评估完成则取消自己的待运行后继。连续三次程序失败或 OOM 会停止该候选自动重试并在报告中标出，以免错误循环消耗 GPU；正常 TIMEOUT 不受此限制。四卡训练改为每个 epoch 保存，以减少短时段退出时丢失的进度；模型和训练超参不变。额外任务登记在 `reports/jobs_flexible.json`，六小时汇报和 GitHub 报告同时追踪它们。

`--time-min` 的较短时段由 Slurm backfill 决定，[官方说明](https://slurm.schedmd.com/sbatch.html#OPT_time-min)。跨节点配置检查已通过，实际 CUDA/NCCL 运行验证需等 GPU 分配；最近资源分配快照见 [resource_snapshot.json](reports/resource_snapshot.json)。

```bash
.env/bin/python scripts/resource_queue.py submit b16 split
.env/bin/python scripts/resource_queue.py submit l16 single
.env/bin/python scripts/resource_queue.py submit l16 split
```

提交器会复用仍在排队或运行的同模型/同形状候选，不因重复调用创建新任务。

新候选在超时或节点故障后自动从检查点恢复。检查状态，或在对应候选不再运行/排队时通过幂等入口恢复：

```bash
.env/bin/python scripts/resource_queue.py submit b16 single
.env/bin/python scripts/resource_queue.py submit l16 single
squeue -u "$USER"
.env/bin/python scripts/report.py
bash scripts/publish.sh
```

训练入口在每个 50K 阶段完成后调用 `scripts/publish.sh` 更新 GitHub；旧 watcher 不作为自动化保证。报告、训练日志、事件和 checkpoint 均在本地保存；大模型文件和 ImageNet 不提交 Git。报告推送使用普通 fast-forward push。

上游 PyTorch 实现固定在 `vendor/JiT`；论文 [Back to Basics](https://arxiv.org/abs/2511.13720)。
