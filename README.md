# Reproduce JiT

ImageNet-1K 上按优先级训练六个官方 JiT 模型 200 epochs：B/16、L/16、B/32、L/32、H/16、H/32。每个模型固定使用作者按模型和分辨率提供的 CFG，并以该 CFG 单独评估；B/16 同时增加 issue #56 的 CFG=3.6/EMA=0.9996 对照。

- [实时状态及六模型结果](reports/REPORT.md)
- [复现前 issue 调查](reports/issue_review.md)
- [逐模型参数配置](configs/)

当前工作目录：`/mnt/home/yliu5/Reproduce_JIT`。

## 训练顺序

```text
B/16, 256², CFG 2.9     issue #56 comparison: CFG 3.6
L/16, 256², CFG 2.4
B/32, 512², CFG 2.9
L/32, 512², CFG 2.5
H/16, 256², CFG 2.2
H/32, 512², CFG 2.3
```

模型队列使用一个 8-GPU H100/H200 Slurm allocation，以上述顺序执行。各配置的像素分辨率、noise scale、H-model dropout、batch 和梯度累积均在 `configs/` 记录；有效 global batch 为 1024。每个最终 FID 在**该模型固定 CFG** 下用 8K 样本选 EMA，再用 50K 样本复核。B/16 的官方 CFG=2.9 是主结果，3.6 对照单独报告。

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

Slurm job script 对应当前集群账号。正式训练申请 8 张 H100/H200、单节点、72 CPU、512 GiB RAM，最长 7 天。数据下载需通过完整 MD5 并解压到 1000 个类别、1,281,167 张原始训练图后才开始训练。

超时或节点故障后，从检查点恢复整个有序队列：

```bash
sbatch scripts/train.sbatch
squeue -u "$USER"
.env/bin/python scripts/report.py
bash scripts/publish.sh
```

后台 watcher 检查六个模型结果；报告更新推送到 GitHub，不强推覆盖远端提交。报告、训练日志和事件、checkpoint 均在本地保存；大模型文件和 ImageNet 不提交 Git。

上游 PyTorch 实现固定在 `vendor/JiT`；论文 [Back to Basics](https://arxiv.org/abs/2511.13720)。
