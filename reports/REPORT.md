# JiT-B/16 ImageNet-256 复现报告

更新时间：2026-09-23T15:44:14.288025+00:00


## 实际状态

已保存训练进度：**0/200 epochs**。

数据：尚未就绪；下载/解压中，详见本地 logs/data-*.log。

GPU 冒烟测试：尚未完成。


```text
97390 jit_b16_200ep PENDING (Dependency)
97387 jit_smoke RUNNING alphagpu12
97338 jit_data RUNNING alphagpu04
```


```text
97338|jit_data|RUNNING|0:0|00:10:38|alphagpu04
97387|jit_smoke|RUNNING|0:0|00:00:44|alphagpu12
97390|jit_b16_200ep|PENDING|0:0|00:00:00|None assigned
```


## 目标和设置


目标是从头训练 JiT-B/16 至 200 epochs，比较论文的 FID-50K **4.37**。

论文原始实现为 JAX/TPU；本工程使用作者公开的 PyTorch/GPU 实现。

完整参数见 [配置](../configs/b16_200ep.json)，来源、成功复现反馈和差异见 [调查报告](issue_review.md)。

8 张 H100/H200，单卡 batch 128，全局 batch 1024；AdamW β=(0.9,0.95)，weight decay=0，实际 LR=2e-4，5 epoch warmup 后恒定。

模型直接预测 x，以 v 空间 MSE 训练；bf16 autocast；保留上游 FP32 attention score、torch.compile 和 t_eps=0.05。

EMA 同时跟踪 0.9996/0.9998/0.9999；CFG 1.0–4.0，步长 0.1，在 8K 样本上搜索；选择后用均衡覆盖 1000 类的 50K 样本确认。

另评估 issue #56 的 EMA=0.9996/CFG=3.6 和 README 默认 EMA=0.9999/CFG=2.9，均用 50K 样本。

预注册随机种子 0；50 步 Heun（末步 Euler），CFG interval [0.1,1.0]；作者定制 torch-fidelity 和 jit_in256_stats.npz。


## 测量结果


**尚无完成 200 epochs 的 FID/IS 结果；不宣称复现成功。** 数据下载、GPU smoke、排队和训练开始均不代表训练完成。


## 证据与限制


数据、checkpoint、TensorBoard 原始事件及完整日志保留在本地，不提交大文件到 GitHub。

checkpoint 使用临时文件原子替换，并保留 100/200 epoch 快照、优化器和各 rank RNG；每 5 epochs 保存一次，中断可能重做至多 5 epochs。

训练采样器与官方一致：DistributedSampler + drop_last，每个 epoch 1251 个 optimizer steps，200 epochs 共 250200 steps。

数据顺序、额外 EMA、独立 DataLoader RNG、监测评估恢复 RNG、软件次版本及硬件差异都可能影响精确数值，因此以真实 FID 和设置差异报告，不保证等于 4.37。

仅单一训练 seed；生成的主评估选参依照论文 8K 搜索规则，不能用多个 50K 结果反向选择最小值作为主结果。
