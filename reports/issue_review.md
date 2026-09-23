# JiT 复现前调查（2026-09-23）

目标：优先训练 JiT-B/16（论文 FID-50K 4.37）和 JiT-L/16（2.79），然后完成 B/32、L/32、H/16、H/32 四个配置，各 200 epochs。[论文表 6](https://arxiv.org/html/2511.13720v1#S5) 报告了全部模型预算和指标。上游 PyTorch 训练脚本经过 8×H200 测试。[官方仓库](https://github.com/LTH14/JiT)。

## 其他人是否复现成功

**有明确的成功自述，但不是附带完整日志/checkpoint 的独立验证。**

- [issue #56](https://github.com/LTH14/JiT/issues/56)：提问者在 8×H100、200 epochs、README 参数下报告 FID 4.72，另有人报告 4.69。
- [Purshow 的成功回复，2026-03-20](https://github.com/LTH14/JiT/issues/56#issuecomment-4096958097)：报告成功复现，并给出 **EMA=0.9996、CFG=3.6**；没有给出精确最终 FID 或完整实验产物。因此这里只将它视为有针对性的复现线索，不把它当成本工程结果。
- JiT 官方 README 对 B/16 ImageNet 256² 的 600-epoch 配方使用训练时评估 CFG **2.9**；L/16 预训练评估示例用 **2.4**。论文说明最佳 EMA 和 CFG 随模型、epoch 变化；issue #56 的 3.6 是独立的 200-epoch 成功者反馈，不是 README 官方默认值。因此本矩阵固定使用 README 针对各模型/分辨率的 CFG，B/16 将 3.6 作为单独 50K 对照。
- [issue #42](https://github.com/LTH14/JiT/issues/42)：作者本人指出 200 和 600 epochs 的最佳 EMA/CFG 差异很大，应搜索两者。
- [issue #41](https://github.com/LTH14/JiT/issues/41)：B/32 的用户在修正评估类别覆盖后得到 FID 4.0928，接近论文 4.02；作者认为约 4.10 的差异在随机波动范围内。这是另一个模型的预训练权重评估，不是 B/16 200 epochs 的从头训练证明。

## 排查到的具体问题及采取的措施

| 风险 | 证据 | 本工程处理 |
|---|---|---|
| `--blr 5e-5` 被当作实际 LR | 上游 `main_jit.py` 按全局 batch/256 缩放；论文表 9 为 2e-4 | 固定 global batch=1024、actual LR=2e-4；启动断言 8 GPUs |
| 默认只用第一个 EMA 评估 | 上游 `engine_jit.py` 硬编码 `ema_params1`；默认 decay1=0.9999 | 三个 EMA 明确按 decay 选择并分别保存，避免只改命令行 EMA 但仍加载错误权重 |
| 不同模型/epoch 的 CFG 不同，且社区的 200-epoch 设置与官方示例不同 | [#56](https://github.com/LTH14/JiT/issues/56)、[#42](https://github.com/LTH14/JiT/issues/42) | 六模型分别固定官方 README 的 CFG：B16 2.9、L16 2.4、B32 2.9、L32 2.5、H16 2.2、H32 2.3；B16 再独立比较 issue #56 的 3.6 |
| 图像灰暗其实是没加载 checkpoint | [#66](https://github.com/LTH14/JiT/issues/66)：用户把文件路径传给需要目录的 resume，导致重新初始化 | 固定读取 run 目录的 `checkpoint-last.pth`，评估时缺失即报错；检查 epoch、EMA、优化器和训练配置 |
| 小样本/单类别 FID 无法与论文比较 | #41 的 800 张和单类评估；更正 1000 类后结果恢复 | 8K 搜索每类 8 张；50K 正式评估每类 50 张；实际输出数严格检查 |
| 删除采样时 `(1-t)` clamp | [#24](https://github.com/LTH14/JiT/issues/24)：作者称 ImageNet 上保留 clipping 有利于 FID | 训练和采样均保留 `t_eps=0.05` |
| x-pred/v-loss 接近 t=1 时不稳定 | [#13](https://github.com/LTH14/JiT/issues/13)：作者强调最小分母约束 | 保留上游分母处理和 bf16，所有 rank 检测非有限 loss 后终止 |
| 把架构消融结果混为完整模型 | [#23](https://github.com/LTH14/JiT/issues/23)：作者解释 bottleneck、RoPE、qk-norm、in-context、CFG interval 的贡献 | 不移除官方 B/16 的这些组件 |
| 改 attention 实现影响数值 | #42 使用 F.sdpa 的实验不是完全相同实现 | 使用上游 FP32 attention-score 路径，不替换 Flash/SDPA |
| 很早期网格图被当作复现失败 | [#71](https://github.com/LTH14/JiT/issues/71)：仅约 2.5 epochs，且 B/32 用在 256 分辨率 | 不能据此断言标准 B/16 训练失败；按完整 200 epochs 和 FID 判断 |

## 锁定配置与上游差异

- 上游 JiT commit：`cbc743a2ada5e9762697da2c83f8c4f8379e8c17`。
- 定制 torch-fidelity commit：`bfe72b995473e4fee9351705d7b005f17442a4ca`。
- Python 3.10、PyTorch 2.5.1+cu124、torchvision 0.20.1、numpy 1.22.4、timm 0.9.12、scipy 1.9.1、tensorboard 2.10.0；其余精确安装版本见 `environment-lock.txt`。
- 原模型/denoiser/crop/lr helper 直接从锁定 submodule 导入，未修改。自建训练管理脚本负责三组 EMA、日志、原子 checkpoint、RNG 恢复与评估选择；不是完全原封不动的官方 main。
- 新增第三组 EMA=0.9998，与论文一致；公开代码默认只维护两组。每模型在固定官方 CFG 下比较三组 EMA。
- 每 40 epochs 做 8K 固定模型 CFG 的监测，不与论文正式 FID 比较；正式结果在每模型官方固定 CFG 下仅搜索三种 EMA，再以 50K 评估。B16 另算 issue #56 对照。
- 使用独立 DataLoader generator，评估前后保存/恢复训练 RNG。种子仍为 0，但随机轨迹不承诺与原始 JAX 或官方 main 逐位一致。
- 实际使用的 H100/H200 型号、设备数、吞吐和显存由运行时记录。初始不替换 Blackwell，避免改变官方 CUDA 12.4 / PyTorch 2.5.1 组合。
- 50K 的正式结果保持该模型的官方 CFG；8K 样本只在同一 CFG 下选 EMA，不使用不同 CFG 的最佳 FID 代替主结果。

## 数据来源与验证

从 [ImageNet 官方训练集地址](https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar) 下载原始 ILSVRC2012 train archive。此地址在本机返回 HTTP 200，支持 Range；大小 147897477120 bytes。

预期 MD5：`1d675b47d978889d74fa0da5fadfb00e`，亦列于 [Intel ImageNet 准备说明](https://github.com/intel/ai-reference-models/blob/main/datasets/imagenet/README.md)。先校验完整 archive，再解压；检查 1000 类与 1,281,167 张图片。不用验证集训练，也不使用预缩放替代集。

原始 API 调查快照保存在本地 `artifacts/issues_index_raw.json` 与 `artifacts/issues_review_raw.json`，方便审计；不把这些社区讨论的全文复制到 GitHub 报告中。
