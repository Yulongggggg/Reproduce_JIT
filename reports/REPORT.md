# JiT 官方配置复现实验报告

更新时间：2026-09-23T17:16:26.089966+00:00

## 当前进度

**已完成 200 epochs：0/6。** 当前训练进度：

- JiT-B/16：0/200 epochs

- JiT-L/16：0/200 epochs

- JiT-B/32：0/200 epochs

- JiT-L/32：0/200 epochs

- JiT-H/16：0/200 epochs

- JiT-H/32：0/200 epochs

GPU smoke：passed，NVIDIA H100 80GB HBM3。

ImageNet 准备中：下载分块 551/551；完整校验和解压尚未完成。


Slurm 队列：
```text
97491 jit_smoke PENDING (Priority)
97390 jit_b16_200ep PENDING (Dependency)
97338 jit_data RUNNING alphagpu04
```

## 预注册设置

依次训练 JiT-B/16、JiT-L/16、B/32、L/32、H/16、H/32；每个模型从头训练 200 epochs，AdamW、实际 LR=2e-4、全局有效 batch=1024、5 epoch warmup、constant LR。

固定 CFG 和分辨率：B/16 256² CFG 2.9（官方；另测 issue #56 的 CFG 3.6、EMA 0.9996）；L/16 256² CFG 2.4；B/32 512² CFG 2.9；L/32 512² CFG 2.5；H/16 256² CFG 2.2；H/32 512² CFG 2.3。CFG interval 均为 [0.1,1.0]，ODE solver 为 50-step Heun。

CFG 取自作者 README 按模型和分辨率给出的训练／评估示例。每个 CFG 固定，不跨值搜索；每个模型只在该 CFG 下用 8K 选择 EMA，再以相同 CFG 和 EMA 计算 FID-50K。

训练超参、批量和累积步数见 [`configs/`](../configs/)。梯度累积保持全局有效 batch=1024。H 型官方 proj dropout=0.2，其余为 0。P_mean=-0.8、P_std=0.8、t_eps=0.05、label drop=0.1、noise scale=res/256。


## FID-50K 对照

| 模型 | 分辨率 | 固定 CFG | 200-epoch 论文 FID | 本次 EMA | 本次 FID-50K | 差值 |

|---|---:|---:|---:|---:|---:|---:|

| JiT-B/16 | 256² | 2.9 | 4.37 | — | — | — |

| JiT-L/16 | 256² | 2.4 | 2.79 | — | — | — |

| JiT-B/32 | 512² | 2.9 | 4.64 | — | — | — |

| JiT-L/32 | 512² | 2.5 | 3.06 | — | — | — |

| JiT-H/16 | 256² | 2.2 | 2.29 | — | — | — |

| JiT-H/32 | 512² | 2.3 | 2.51 | — | — | — |


**训练尚未开始；目前没有本次实验的 FID。**


## 复现证据与边界

上游 JiT 与定制 torch-fidelity 使用锁定 submodule；原模型、denoiser、loss 和 attention 保持上游实现。环境锁、GPU smoke 和数据验证记录见本目录。

训练保存 checkpoint、三组 EMA、优化器和各 rank RNG，每 5 epochs 保存一次；执行顺序为 B/16、L/16 优先，随后其余四种。

200 epoch 是目标训练预算；FID 使用 1000 类均衡的 50K 样本。8K 只用于选 EMA，不代替正式结果。
