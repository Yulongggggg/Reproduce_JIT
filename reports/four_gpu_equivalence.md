# Four-GPU JiT-B/16 and JiT-L/16 run

The paper's [Table 9](https://arxiv.org/html/2511.13720) specifies a global training batch of 1024, learning rate 2e-4, five warmup epochs, constant learning rate, and 200 epochs for ablations. The [official PyTorch example](https://github.com/LTH14/JiT/blob/main/README.md) uses eight GPUs with per-GPU batch 128 for B/16; its [argument parser](https://github.com/LTH14/JiT/blob/main/main_jit.py) defines batch size per GPU and scales the learning rate from the resulting global batch. The official runner does not provide gradient accumulation, so simply changing its process count to four would halve the batch and alter the default learning rate.

Our four-GPU runner averages gradients over more microbatches before each optimizer and EMA update:

| Model | GPUs | Batch per GPU | Accumulation | Global batch | Learning rate | CFG |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B/16, original run | 8 | 128 | 1 | 1024 | 2e-4 | 2.9 |
| B/16, four-GPU run | 4 | 128 | 2 | 1024 | 2e-4 | 2.9 |
| L/16, original run | 8 | 64 | 2 | 1024 | 2e-4 | 2.4 |
| L/16, four-GPU run | 4 | 64 | 4 | 1024 | 2e-4 | 2.4 |

All other training, EMA, and sampling parameters come from the corresponding eight-GPU config. Four GPUs preserve the number of images per optimizer update and the number of updates per epoch. Random draws and floating-point reduction order change with process count, so bitwise-identical weights or identical FID are not guaranteed. The [B/16 reproduction in issue #56](https://github.com/LTH14/JiT/issues/56) used eight H100s; [issue #29](https://github.com/LTH14/JiT/issues/29) used one A6000 on CIFAR-10, which does not establish ImageNet FID equivalence on fewer GPUs. No four-GPU ImageNet FID reproduction was found in the reviewed issues.

The four-GPU jobs use separate `runs/b16_4gpu_200ep` and `runs/l16_4gpu_200ep` checkpoints. The first Slurm job is 101979; the complete dependent chain is recorded in [jobs_4gpu.json](jobs_4gpu.json). Each priority-QOS segment has a 12-hour limit; the next segment becomes eligible only if the previous one ends unsuccessfully (including a time limit). Both 200-epoch models are evaluated with their model-specific CFG values. The existing eight-GPU chain remains queued separately.
