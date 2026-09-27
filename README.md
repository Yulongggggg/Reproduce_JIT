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

当前运行 4 卡 B/16 → L/16 队列，原 8 卡队列保留为独立实验，输出目录分开。B/16 每卡 batch=128、累积 2 次，L/16 每卡 batch=64、累积 4 次，均为有效 batch=1024。其他四个模型仅保留配置，暂不进入当前队列。

## 自动 FID-50K

每个模型在 200 epoch 后执行：

1. 固定 B/16 CFG=2.9、L/16 CFG=2.4，用 8K 选 EMA，计算 FID-50K。
2. 按[论文 Appendix A](https://arxiv.org/html/2511.13720v2#A1)，在 CFG=1.0–4.0（步长 0.1）和三个 EMA（0.9996/0.9998/0.9999）间做 93 组 8K 评估，再用选出的参数计算 FID-50K。
3. 正式评估均生成 1000 类 × 每类 50 张，采用作者的 FID 统计文件、torch-fidelity、50-step Heun 和 CFG interval [0.1,1.0]；记录统计文件 hash、后端版本和 checkpoint 标识。
4. 自动更新、提交并推送报告；对照[论文 Table 6](https://arxiv.org/html/2511.13720v2#S5.T6) 的 **200 epoch** B/16=4.37、L/16=2.79。样本数、轮数或协议不匹配时不计算差值。论文未报告的中间轮数不填造基准。

93 组搜索共需生成 744K 图片，评估时间需另计；每组结果单独保存，超时后复用。8K 是监测/选参分数，正式对照全部使用 50K。固定 CFG 表不声称与论文的搜索流程相同。报告推送失败不会中断训练，错误写入各运行目录的 `report_publish.json`。

## 运行与恢复

```bash
git clone --recurse-submodules https://github.com/Yulongggggg/Reproduce_JIT.git
cd Reproduce_JIT
bash scripts/install.sh
mkdir -p logs
data_job=$(sbatch --parsable scripts/data.sbatch)
smoke_job=$(sbatch --parsable scripts/smoke.sbatch)
sbatch --dependency=afterok:${data_job}:${smoke_job} scripts/train.sbatch
```

Slurm job script 对应当前集群账号。当前四卡脚本为 `scripts/train4.sbatch`，申请 4 张 H100/H200、40 CPU、256 GiB RAM，每次最多 12 小时，已提交自动续跑链（见 `reports/jobs_4gpu.json`，不要重复提交）。数据已通过完整 MD5，并解压为 1000 类、1,281,167 张训练图片。

超时或节点故障后，从检查点恢复整个有序队列：

```bash
sbatch scripts/train.sbatch
squeue -u "$USER"
.env/bin/python scripts/report.py
bash scripts/publish.sh
```

训练入口在每个 50K 阶段完成后调用 `scripts/publish.sh` 更新 GitHub；旧 watcher 不作为自动化保证。报告、训练日志、事件和 checkpoint 均在本地保存；大模型文件和 ImageNet 不提交 Git。报告推送使用普通 fast-forward push。

上游 PyTorch 实现固定在 `vendor/JiT`；论文 [Back to Basics](https://arxiv.org/abs/2511.13720)。
