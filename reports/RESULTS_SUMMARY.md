# JiT reproduction: results summary

ImageNet-1K, 256×256, 200 training epochs, official JiT code. Both models reproduce the paper's FID-50K.

| Model | Paper FID-50K | Our FID-50K | Difference | Selected EMA / CFG | Inception Score | Completed |
|---|---|---|---|---|---|---|
| JiT-B/16 | 4.37 | **4.319** | −0.051 | 0.9998 / 3.6 | 259.1 | 10 Oct 2026 |
| JiT-L/16 | 2.79 | **2.747** | −0.043 | 0.9998 / 2.6 | 282.4 | 16 Oct 2026 |

Paper values: Li & He, *Back to Basics: Let Denoising Generative Models Denoise* ([arXiv 2511.13720](https://arxiv.org/abs/2511.13720)), Table 6, 200 epochs.

## How the numbers were computed

We follow the paper's Appendix A evaluation protocol:

1. For each EMA decay in {0.9996, 0.9998, 0.9999} and CFG scale from 1.0 to 4.0 in steps of 0.1 (93 settings), generate 8,000 class-balanced images and compute FID.
2. With the setting of lowest 8K FID, generate 50,000 images (50 per class) and compute FID-50K.

Sampling uses Heun with 50 steps and a CFG interval of [0.1, 1.0], as in the official code. FID uses the official ImageNet-256 reference statistics (`jit_in256_stats.npz`, sha256 `41204672…`) and the LTH14 fork of torch-fidelity.

## Additional FID-50K results

These use the fixed CFG from the official README instead of the search. They are reported for comparison only.

| Model | EMA | CFG | FID-50K | Inception Score |
|---|---|---|---|---|
| JiT-B/16 | 0.9998 | 2.9 (README) | 4.712 | 218.3 |
| JiT-B/16 | 0.9996 | 3.6 | 4.338 | 255.2 |
| JiT-L/16 | 0.9998 | 2.4 (README) | 2.760 | 265.8 |

An independent public reproduction of B/16 with the official code ([JiT issue #56](https://github.com/LTH14/JiT/issues/56)) reports 4.72 at CFG 2.9. This matches our 4.712.

## Training setup

- **Code**: official JiT repository at commit `cbc743a2`, unmodified (`vendor/JiT`). Our wrapper adds checkpoint resume, job scheduling and evaluation only.
- **Data**: official ILSVRC2012 training set, 1,281,167 images in 1,000 classes. The archive checksum matches torchvision's.
- **Optimization**: official per-model configuration, with an effective batch of 1,024 for both models.
- **JiT-B/16**: 4 × NVIDIA H100 80GB, 128 images per GPU, 2 accumulation steps.
- **JiT-L/16**: epochs 0–12 ran on 4 GPUs. Training then resumed on 8 GPUs with 64 images per GPU and 2 accumulation steps. The final segment ran on 8 × NVIDIA H200.
  - The 4 → 8 GPU switch preserved the model, optimizer, all three EMA copies, the epoch and the global step.
  - Data order and floating-point reduction order differ from an uninterrupted run, so the result is not bitwise-identical to one.
- **Scope**: neither model has been trained beyond 200 epochs. The planned 600-epoch extension of B/16 was cancelled.

## Evidence in this repository

- Per-model results with every evaluated setting:
  - [`b16_results.json`](b16_results.json)
  - [`l16_results.json`](l16_results.json)
- Paper comparison table: [`comparison_50k.csv`](comparison_50k.csv)
- Each of the 93 search evaluations and each final 50K evaluation: [`results/b16_4gpu_200ep/evaluations/`](results/b16_4gpu_200ep/evaluations/) and [`results/l16_4gpu_200ep/evaluations/`](results/l16_4gpu_200ep/evaluations/)
- 8 × 8 sample grids from the final 50K runs: `results/*/samples/`
- Training loss per epoch:
  - [`training.csv`](training.csv)
  - [`loss.png`](loss.png)
- Evaluation protocol definition: [`evaluation_protocol.json`](evaluation_protocol.json)
- Full running log of the reproduction, in Chinese: [`REPORT.md`](REPORT.md)
