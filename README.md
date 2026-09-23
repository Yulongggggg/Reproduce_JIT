# Reproduce JiT

JiT-B/16 / ImageNet 256×256 / 200 epochs 从头复现。目标参考 FID-50K **4.37**；尚未跑完时不宣称复现成功。

- [进度及结果报告](reports/REPORT.md)
- [复现前 issue 调查和配置差异](reports/issue_review.md)
- [锁定训练配置](configs/b16_200ep.json)

本地工作目录：`/mnt/home/yliu5/Reproduce_JIT`。

## 运行

```bash
git clone --recurse-submodules https://github.com/Yulongggggg/Reproduce_JIT.git
cd Reproduce_JIT
bash scripts/install.sh
mkdir -p logs
data_job=$(sbatch --parsable scripts/data.sbatch)
smoke_job=$(sbatch --parsable scripts/smoke.sbatch)
sbatch --dependency=afterok:${data_job}:${smoke_job} scripts/train.sbatch
```

Slurm 模板对应当前集群账号；换集群需修改 account/partition/QOS。训练申请单节点 8×H100/H200、72 CPU、512 GiB RAM，最长 7 天。正式运行前必须完整数据校验和 GPU smoke 成功。训练结束后自动执行论文的 EMA/CFG 搜索、正式 FID-50K 及对照评估。

```bash
# 从最近 checkpoint 继续到总共 200 epochs，再继续未完成评估
sbatch scripts/train.sbatch
# 检查作业
squeue -u "$USER"
# 更新本地报告，或将报告提交到 GitHub
.env/bin/python scripts/report.py
bash scripts/publish.sh
```

`scripts/watch.py` 监测本次提交的作业、更新 GitHub 报告；对节点失败、抢占、超时最多续交 3 次，普通代码错误不会自动无限重试。报告同步拒绝 force-push。

数据和 checkpoint 不进 Git：`data/imagenet`、`runs/b16_200ep`、`logs`。每 5 epochs 保存模型、3 组 EMA、优化器、各 rank RNG，另外保留 100/200 epoch checkpoint。TensorBoard 在 `runs/b16_200ep/tensorboard`。

## 上游

[论文](https://arxiv.org/abs/2511.13720) · [官方 JiT](https://github.com/LTH14/JiT) · [定制 torch-fidelity](https://github.com/LTH14/torch-fidelity)

保留上游代码在各自 submodule 中，许可见对应目录的 LICENSE。复现管理脚本见本仓库 LICENSE。
