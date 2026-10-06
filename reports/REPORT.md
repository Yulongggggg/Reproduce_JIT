# JiT 官方配置复现实验报告

更新时间：2026-10-06T06:03:45.188406+00:00

## 当前进度

**已完成训练与最终评估：0/2。** 目录沿用原四卡路径；实际训练卡数以运行元数据为准。训练轮数取自逐轮日志，每轮保存检查点，异常退出后从检查点恢复：

**当前候选任务在排队，没有正在运行的 JiT 训练作业。** 续跑从检查点恢复，尚未保存的日志轮数需要重跑。

用户最新安排：JiT 训练只使用 standard QoS。B/16、L/16 各申请单节点 8 张 H100/H200，单次 24–48 小时；旧四卡 priority 和八卡 long 候选已撤销且不再自动恢复。所有候选共享同模型文件锁，从现有检查点续跑。其他项目不在本次调整范围。

追加安排（2026-10-05）：保留上述两个 standard8 排队任务，另提交一个一次性 priority 四卡任务（作业 120736），只把 B/16 从第 184 轮检查点训练到 200 轮，沿用前 184 轮的四卡配置（每卡 batch=128、累积 2 次、有效 batch=1024）。该任务不做最终评估，也不自动续跑；B/16 的 FID-50K 仍由 standard8 任务完成。若 B/16 的 standard8 任务在它运行期间获得资源，会等它结束再接手，不放弃已分配的资源。

八卡 B/16 每卡 batch=128、累积 1 次；L/16 每卡 batch=64、累积 2 次，均保持有效 batch=1024。四卡转八卡保留模型、优化器、三组 EMA、epoch 和 global step；保留旧 rank RNG，并为新增 rank 设置独立种子。数据分片、随机轨迹和归约顺序会变化，不声称逐位一致，也不把它标为从头八卡训练。

扩展计划：先完成 200 epoch 的固定 CFG 和完整论文流程 FID-50K，再判断是否延长到 600 epoch。质量阈值尚未最终确定，目前没有提交或自动启动 600 epoch；200 epoch 完成后定时器继续汇报，等待扩展决定。

JiT-B/16 最近一次训练元数据记录为 4 卡。

JiT-L/16 最近一次训练元数据记录为 4 卡。

**跨节点启动故障：B/16 作业 107750（FAILED）；L/16 作业 107752（FAILED）。** 两个任务在美东 2026-09-29 14:37 先后获分配，各用 4 张 H200（2 节点各 2 卡），均在初始 NCCL barrier 报 `ibv_modify_qp: Invalid argument errno 22`，尚未进入训练，未推进检查点。已仅为双节点入口配置 `NCCL_IB_DISABLE=1`、`NCCL_NET=Socket`，双节点入口保留真实四卡 all-reduce 检查；当前八卡单节点任务不使用此入口。故障证据与配置范围见 [记录](multinode_network_issue.json)。

JiT-B/16 TCP 替代方案尚无真实 GPU 验证结果，不能宣称故障已解决。

JiT-L/16 TCP 替代方案尚无真实 GPU 验证结果，不能宣称故障已解决。

自动汇报：每 6 小时在本对话和 GitHub 更新（美东 02:30, 08:30, 14:30, 20:30），状态 `active`；下一次计划时间：2026-10-06T02:30:00-04:00。定时器所在登录主机需保持运行。

- JiT-B/16：训练历史日志最高完成 184/200 epochs；可恢复检查点为第 184 轮。

- JiT-L/16：训练历史日志最高完成 12/200 epochs；可恢复检查点为第 12 轮。

JiT-B/16 最近 5 个已完成 epoch 平均 5.8 分钟；仅在相同吞吐下，剩余训练约 1.5 小时，不含评估、重试和排队。改变卡数或节点通信方式后的速度需实测，不能直接沿用此前耗时。

JiT-B/16 第 160 epoch 中途评估：FID-8000=8.9342，CFG=2.9，EMA=0.9996。这是中途 8K 指标，不能作为最终 FID-50K 结果。

JiT-L/16 最近 5 个已完成 epoch 平均 14.2 分钟；仅在相同吞吐下，剩余训练约 44.3 小时，不含评估、重试和排队。改变卡数或节点通信方式后的速度需实测，不能直接沿用此前耗时。

此前取消的队列维持取消；新八卡 standard 候选登记在 jobs_flexible.json，最新切换和取消记录见 [qos_switch_standard.json](qos_switch_standard.json)。

资源调度：唯一启用的候选类型为 standard8，B/16 与 L/16 可独立运行，每任务 world_size=8、有效 batch=1024。自动续跑同样使用 standard QoS；同模型文件锁防止并发写检查点，priority 和 long 的提交与启动入口已禁用。

资源检查快照（2026-09-29T18:31:45.334160+00:00）：28 台 GPU 节点共 224 卡，其中 224 卡已被 Slurm 分配。这是分配计数，不是 GPU 利用率；MIXED 节点可能仅 CPU 有空闲。详细资源和候选状态见 [resource_snapshot.json](resource_snapshot.json)。

GPU smoke：passed，NVIDIA H100 80GB HBM3。

ImageNet 完整数据已校验、解压：1,281,167 images，1000 classes。


Slurm 队列：
```text
120736 jit_b16_priority4 PENDING (Priority)
111416 jit_l16_standard8 PENDING (Priority)
111415 jit_b16_standard8 PENDING (Priority)
```

流水线记录：2026-10-06 02:03 EDT 自动检查：目前没有 JiT 训练作业运行；候选作业 111415 正在排队（(Priority)）。 训练历史最大轮数和可恢复检查点分别列出。

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
