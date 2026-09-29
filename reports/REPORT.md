# JiT 官方配置复现实验报告

更新时间：2026-09-29T06:30:38.689722+00:00

## 当前进度

**已完成训练与最终评估：0/2。** 当前展示独立的 4 卡实验。训练轮数取自逐轮日志；此前每 5 轮保存，新增资源候选后四卡训练每轮保存，异常退出后从检查点恢复：

自动汇报：每 6 小时在本对话和 GitHub 更新（美东 02:30, 08:30, 14:30, 20:30），状态 `active`；下一次计划时间：2026-09-29T08:30:00-04:00。定时器所在登录主机需保持运行。

- JiT-B/16：训练历史日志最高完成 178/200 epochs；可恢复检查点为第 178 轮。

- JiT-L/16：训练历史日志最高完成 0/200 epochs；可恢复检查点为第 0 轮。

JiT-B/16 最近 5 个 epoch 平均 5.9 分钟；剩余训练约 2.1 小时，不含评估、重试和排队。

JiT-B/16 第 160 epoch 中途评估：FID-8000=8.9342，CFG=2.9，EMA=0.9996。这是中途 8K 指标，不能作为最终 FID-50K 结果。

按用户最新要求，全部训练任务均为总计四卡。遗留八卡待运行队列已取消；取消记录见 [jobs_8gpu_retired.json](jobs_8gpu_retired.json)。

资源调度：B/16 与 L/16 可独立运行；候选包括单节点 4 卡和双节点各 2 卡，均保持 world_size=4、有效 batch=1024。每模型使用文件锁防止并发写同一检查点，候选申请 3–12 小时以利用短空档；失败/超时自动提交的后继从检查点续跑。候选不改变模型、优化器、CFG 或评估协议。

资源检查快照（2026-09-29T02:28:15.172143+00:00）：28 台 GPU 节点共 224 卡，其中 223 卡已被 Slurm 分配。这是分配计数，不是 GPU 利用率；MIXED 节点可能仅 CPU 有空闲。详细资源和候选状态见 [resource_snapshot.json](resource_snapshot.json)。双节点任务已通过 Slurm 配置检查，CUDA/NCCL 实际运行仍待分配资源验证。

GPU smoke：passed，NVIDIA H100 80GB HBM3。

ImageNet 完整数据已校验、解压：1,281,167 images，1000 classes。


Slurm 队列：
```text
107752 jit_l16_split PENDING (Priority)
107750 jit_b16_split PENDING (Priority)
107751 jit_l16_single PENDING (Priority)
101994 jit_b16_l16_4gpu PENDING (Dependency)
101993 jit_b16_l16_4gpu PENDING (Dependency)
101992 jit_b16_l16_4gpu PENDING (Dependency)
101991 jit_b16_l16_4gpu PENDING (Dependency)
101990 jit_b16_l16_4gpu PENDING (Dependency)
101989 jit_b16_l16_4gpu PENDING (Dependency)
101988 jit_b16_l16_4gpu PENDING (Dependency)
101987 jit_b16_l16_4gpu PENDING (Dependency)
101986 jit_b16_l16_4gpu PENDING (Dependency)
101985 jit_b16_l16_4gpu PENDING (Dependency)
101984 jit_b16_l16_4gpu PENDING (Dependency)
101983 jit_b16_l16_4gpu RUNNING alphagpu20
```

流水线记录：2026-09-29 02:30 EDT 自动检查：正在运行：101983 (alphagpu20)。 训练历史最大轮数和可恢复检查点分别列出。

## 预注册设置

当前按用户要求优先训练 JiT-B/16、JiT-L/16，各 200 epochs；B/16 从已有检查点续跑，L/16 独立排队。AdamW、实际 LR=2e-4、全局有效 batch=1024、5 epoch warmup、constant LR。4 张卡时 B/16 每卡 batch 128、累积 2 次；L/16 每卡 batch 64、累积 4 次。

固定 CFG 和分辨率：B/16 256² CFG 2.9（官方；另测 issue #56 的 CFG 3.6、EMA 0.9996）；L/16 256² CFG 2.4。CFG interval 均为 [0.1,1.0]，ODE solver 为 50-step Heun。

固定 CFG 结果沿用作者 README：在该 CFG 下用 8K 选择 EMA，再生成 50K。另按论文 Appendix A 搜索 CFG=1.0–4.0（步长 0.1）与 EMA={0.9996,0.9998,0.9999} 共 93 组，每组 8K；选出最低 FID 对应参数后生成 50K 并计算 FID。固定 CFG 的结果和论文搜索结果分别列出。

训练超参、批量和累积步数见 [`configs/`](../configs/)。B/L 的 dropout 为 0，P_mean=-0.8、P_std=0.8、t_eps=0.05、label drop=0.1、noise scale=1。4 卡等效设置和数值复现的边界见 [说明](four_gpu_equivalence.md)。


## 同 epoch、论文评估流程的 FID-50K 对照

官方基准来自 [论文 Table 6](https://arxiv.org/html/2511.13720v2#S5.T6)，均为 200 epochs、256²、FID-50K。本文 40/80/120/160 epoch 的 8K 监测值不参与该表；论文未提供这些轮数的可直接对照值。

未找到 B/16、L/16 可直接对照的官方 100 epoch FID-50K；100 epoch checkpoint 保留，但正式对照选择论文明确公布的 200 epoch。没有真实 50K 结果时，不判断是否已接近官方。

两边使用 1000 类、每类 50 张生成图片，对 ImageNet 训练集计算 FID。复现使用作者发布的 `jit_in256_stats.npz`、锁定的 torch-fidelity、50-step Heun、CFG interval [0.1,1.0]。评估 JSON 记录统计文件 SHA-256、后端 commit、CFG、EMA、epoch 和 checkpoint 标识。论文数值来自原实现；按公开 PyTorch 代码对齐评估流程，不承诺逐位一致。

| 模型 | epoch（双方） | 分辨率 | 官方 FID-50K | 本次 FID-50K | 本次 CFG / EMA | 差值 |
|---|---:|---:|---:|---:|---|---:|
| JiT-B/16 | 200 | 256² | 4.37 | 待评估 | — | — |
| JiT-L/16 | 200 | 256² | 2.79 | 待评估 | — | — |


### README 固定 CFG 的 FID-50K

此表保留指定 CFG 的结果。论文的最优 EMA/CFG 随训练轮数变化，因此本表不计算论文差值；完整搜索结果见上表。

| 模型 | epoch | 固定 CFG | 选中 EMA | FID-50K |
|---|---:|---:|---:|---:|
| JiT-B/16 | 200 | 2.9 | — | 待评估 |
| JiT-L/16 | 200 | 2.4 | — | 待评估 |


![训练曲线](loss.png)


## 复现证据与边界

上游 JiT 与定制 torch-fidelity 使用锁定 submodule；原模型、denoiser、loss 和 attention 保持上游实现。环境锁、GPU smoke 和数据验证记录见本目录。

训练保存 checkpoint、三组 EMA、优化器和各 rank RNG。此前每 5 epochs 保存，增加短时段候选后四卡训练每 epoch 保存；B/16、L/16 可独立并行。

200 epoch 是目标训练预算；FID 使用 1000 类均衡的 50K 样本。8K 用于训练监测和 EMA/CFG 选择，不代替正式结果。

2026-09-27 加入完整论文评估流程；训练脚本在作业启动时加载，排队中的自动续跑任务会读取更新后的版本，在 200 epoch 触发最终评估。

每个 50K 阶段完成后自动生成此报告并尝试提交、推送到 GitHub。单组评估结果可断点复用；推送失败记录在运行目录的 `report_publish.json`，不会中断训练。
