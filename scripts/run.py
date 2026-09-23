"""Original JiT model/loss/attention; explicit EMA selection and auditable runs."""
import argparse
import copy
from contextlib import contextmanager, nullcontext
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'vendor/JiT'))
import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import torch_fidelity
from denoiser import Denoiser
from util.crop import center_crop_arr
import util.lr_sched as lr_sched
import util.misc as misc


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def append_json(path, value):
    with Path(path).open('a') as f:
        f.write(json.dumps(value) + '\n')


def init_distributed(args):
    args.rank = int(os.environ['RANK'])
    args.world_size = int(os.environ['WORLD_SIZE'])
    args.gpu = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(args.gpu)
    # Rank zero computes FID while other ranks wait; allow slow shared-storage evaluation.
    dist.init_process_group('nccl', init_method='env://', rank=args.rank,
                            world_size=args.world_size, timeout=datetime.timedelta(hours=1))
    dist.barrier(device_ids=[args.gpu])
    misc.setup_for_distributed(args.rank == 0)


def rng_state():
    return {'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state(),
            'numpy': np.random.get_state(), 'python': random.getstate()}


def restore_rng(state):
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state(state['cuda'])
    np.random.set_state(state['numpy'])
    random.setstate(state['python'])


@contextmanager
def evaluation_rng(seed):
    state = rng_state()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    try:
        yield
    finally:
        restore_rng(state)


def save_checkpoint(model, optimizer, args, epoch, global_step, loader_generator):
    states = [None] * dist.get_world_size()
    dist.all_gather_object(states, {**rng_state(), 'loader': loader_generator.get_state()})
    if misc.is_main_process():
        checkpoint = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                      'epoch': epoch, 'global_step': global_step, 'config': vars(args),
                      'rng_per_rank': states, 'world_size': dist.get_world_size()}
        for decay, params in model.emas.items():
            # Buffer values are part of state_dict and are retained as in upstream.
            state = dict(model.state_dict())
            state.update({name: value for (name, _), value in zip(model.named_parameters(), params)})
            checkpoint[f'ema_{decay}'] = state
        path = Path(args.output_dir) / 'checkpoint-last.pth'
        tmp = path.with_suffix('.tmp')
        torch.save(checkpoint, tmp)
        tmp.replace(path)
        if epoch + 1 in (100, 200):
            milestone = path.with_name(f'checkpoint-{epoch+1:03d}.pth')
            if not milestone.exists():
                os.link(path, milestone)
        atomic_json(Path(args.output_dir) / 'progress.json', {
            'status': 'trained' if epoch + 1 == args.epochs else 'training',
            'completed_epochs': epoch + 1, 'target_epochs': args.epochs,
            'global_step': global_step, 'checkpoint': str(path),
            'updated_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()})
    dist.barrier()


@torch.no_grad()
def evaluate(model, args, decay, cfg, count, phase, epoch):
    rank, world = misc.get_rank(), misc.get_world_size()
    assert count % args.class_num == 0
    output = Path(args.output_dir)
    key = f'{phase}-ep{epoch:03d}-ema{decay}-cfg{cfg:.1f}-n{count}'
    result_file = output / 'evaluations' / (key + '.json')
    if result_file.exists():
        return json.loads(result_file.read_text())
    folder = output / 'generated' / key
    if rank == 0:
        folder.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    backup = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    for target, source in zip(model.parameters(), model.emas[str(decay)]):
        target.copy_(source)
    model.eval()
    model.cfg_scale = cfg
    labels_per_class = count // args.class_num
    start_time = time.monotonic()
    with evaluation_rng(args.seed + rank):
        for batch_start in range(0, count, args.gen_bsz * world):
            start = batch_start + rank * args.gen_bsz
            indices = torch.arange(start, start + args.gen_bsz, device='cuda')
            labels = (indices // labels_per_class).clamp(max=args.class_num - 1)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                samples = model.generate(labels)
            pixels = ((samples + 1) / 2).cpu().numpy()
            for j in range(min(args.gen_bsz, max(0, count - start))):
                array = np.round(np.clip(pixels[j].transpose(1, 2, 0) * 255, 0, 255)).astype(np.uint8)
                Image.fromarray(array).save(folder / f'{start+j:05d}.png')
            if rank == 0:
                print(f'{key}: generated {min(batch_start + args.gen_bsz * world, count)}/{count}', flush=True)
        dist.barrier()
        result = None
        if rank == 0:
            actual = len(list(folder.glob('*.png')))
            assert actual == count, (actual, count)
            metrics = torch_fidelity.calculate_metrics(
                input1=str(folder), input2=None,
            fid_statistics_file=str(ROOT / 'vendor/JiT/fid_stats' / f'jit_in{args.img_size}_stats.npz'),
                cuda=True, isc=True, fid=True, kid=False, prc=False, verbose=False)
            result = {'phase': phase, 'completed_epochs': epoch, 'ema': decay, 'cfg': cfg,
                      'num_images': count, 'seed': args.seed, 'world_size': world,
                      'gen_bsz': args.gen_bsz, 'seconds': time.monotonic() - start_time,
                      **{k: float(v) for k, v in metrics.items()}}
            # Retain fixed samples across classes for the final report, then reclaim PNG space.
            if phase == 'final':
                sample_dir = output / 'samples' / key
                sample_dir.mkdir(parents=True, exist_ok=True)
                for idx in range(0, count, max(1, count // 64)):
                    shutil.copy2(folder / f'{idx:05d}.png', sample_dir)
            atomic_json(result_file, result)
            print(json.dumps(result), flush=True)
            shutil.rmtree(folder)
        objects = [result]
        dist.broadcast_object_list(objects, src=0)
    model.load_state_dict(backup)
    model.cfg_scale = args.cfg
    torch.cuda.empty_cache()
    return objects[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(ROOT / 'configs/b16_200ep.json'))
    parser.add_argument('--mode', choices=['train', 'evaluate', 'smoke'], default='train')
    parser.add_argument('--smoke-steps', type=int, default=5)
    cli = parser.parse_args()
    cfg = json.loads(Path(cli.config).read_text())
    args = argparse.Namespace(**cfg, dist_on_itp=False, dist_url='env://', distributed=True)
    args.data_path = str(ROOT / args.data_path)
    args.output_dir = str(ROOT / args.output_dir)
    init_distributed(args)
    rank, world = misc.get_rank(), misc.get_world_size()
    if cli.mode != 'smoke':
        assert world == 8 and args.batch_size * world * args.grad_accumulation == 1024
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    random.seed(args.seed + rank)
    torch.backends.cudnn.benchmark = True
    torch._dynamo.config.cache_size_limit = 128
    torch._dynamo.config.optimize_ddp = False
    model = Denoiser(args).cuda()
    ddp = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    model.emas = {str(decay): [p.detach().clone() for p in model.parameters()]
                  for decay in args.ema_decays}
    optimizer = torch.optim.AdamW(misc.add_weight_decay(model, args.weight_decay),
                                 lr=args.lr, betas=(0.9, 0.95))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if cli.mode == 'smoke':
        times, losses = [], []
        for step in range(cli.smoke_steps):
            start = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            for microstep in range(args.grad_accumulation):
                x = torch.rand(args.batch_size, 3, args.img_size, args.img_size, device='cuda') * 2 - 1
                y = torch.arange(args.batch_size, device='cuda') % args.class_num
                sync_context = nullcontext() if microstep + 1 == args.grad_accumulation else ddp.no_sync()
                with sync_context:
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        loss = ddp(x, y)
                    assert torch.isfinite(loss), loss
                    (loss / args.grad_accumulation).backward()
            optimizer.step()
            update_emas(model)
            torch.cuda.synchronize()
            losses.append(loss.item())
            times.append(time.monotonic() - start)
        # Exercise the actual 50-step Heun generation path without calculating a fake FID.
        model.eval()
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            samples = model.generate(torch.tensor([0, 999], device='cuda'))
        assert samples.shape == (2, 3, 256, 256) and torch.isfinite(samples).all()
        # Checkpoint round-trip, including optimizer and three EMA buffers.
        generator = torch.Generator().manual_seed(args.seed + rank)
        original_output = args.output_dir
        safe_model = args.model.replace('/', '-').replace('JiT-', '').lower()
        args.output_dir = str(ROOT / f'runs/smoke_{safe_model}')
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        save_checkpoint(model, optimizer, args, 0, cli.smoke_steps, generator)
        restored = torch.load(Path(args.output_dir) / 'checkpoint-last.pth', map_location='cpu', weights_only=False)
        for decay in args.ema_decays:
            assert f'ema_{decay}' in restored
        assert restored['global_step'] == cli.smoke_steps
        args.output_dir = original_output
        if rank == 0:
            result = {'status': 'passed', 'purpose': 'synthetic GPU pipeline smoke test; NOT training or FID',
                      'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
                      'world_size': world, 'batch_per_gpu': args.batch_size,
                      'step_seconds': times, 'losses': losses,
                      'max_memory_gib': torch.cuda.max_memory_allocated() / 1024**3,
                      'parameters': sum(p.numel() for p in model.parameters()),
                      'sampling': '50-step Heun, finite outputs', 'checkpoint': 'round-trip passed'}
            atomic_json(ROOT / f'reports/smoke_{safe_model}.json', result)
            if args.model == 'JiT-B/16':
                atomic_json(ROOT / 'reports/smoke.json', result)
            print(json.dumps(result), flush=True)
        dist.destroy_process_group()
        return
    generator = torch.Generator().manual_seed(args.seed + rank)
    path = output / 'checkpoint-last.pth'
    start_epoch, global_step = 0, 0
    if path.exists():
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        assert checkpoint['world_size'] == world, 'Changing GPU count changes batch/RNG semantics'
        for key, value in cfg.items():
            if key not in ('data_path', 'output_dir', 'num_workers'):
                assert checkpoint['config'][key] == value, f'Resume config mismatch: {key}'
        model.load_state_dict(checkpoint['model'])
        for decay, params in model.emas.items():
            state = checkpoint[f'ema_{decay}']
            for (name, _), target in zip(model.named_parameters(), params):
                target.copy_(state[name])
        optimizer.load_state_dict(checkpoint['optimizer'])
        restore_rng(checkpoint['rng_per_rank'][rank])
        generator.set_state(checkpoint['rng_per_rank'][rank]['loader'])
        start_epoch, global_step = checkpoint['epoch'] + 1, checkpoint['global_step']
        del checkpoint
        print(f'Resumed {path}, completed epochs={start_epoch}, optimizer steps={global_step}', flush=True)
    elif cli.mode == 'evaluate':
        raise FileNotFoundError(f'Required trained checkpoint absent: {path}')
    if cli.mode == 'train':
        manifest = json.loads((Path(args.data_path) / 'manifest.json').read_text())
        assert manifest['status'] == 'ready' and manifest['images'] == 1281167
        transform = transforms.Compose([transforms.Lambda(lambda img: center_crop_arr(img, args.img_size)),
                                        transforms.RandomHorizontalFlip(), transforms.PILToTensor()])
        dataset = datasets.ImageFolder(Path(args.data_path) / 'train', transform=transform)
        assert len(dataset) == 1281167 and len(dataset.classes) == 1000
        assert dataset.classes == sorted(manifest['classes'])
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True)
        loader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=args.batch_size,
                    num_workers=args.num_workers, pin_memory=True, drop_last=True, generator=generator)
        assert len(loader) % args.grad_accumulation == 0
        steps_per_epoch = len(loader) // args.grad_accumulation
        if rank == 0:
            atomic_json(output / 'run_metadata.json', {
                'config': vars(args), 'world_size': world, 'microsteps_per_epoch': len(loader),
                'steps_per_epoch': steps_per_epoch, 'grad_accumulation': args.grad_accumulation,
                'dataset_images': len(dataset), 'images_per_epoch': len(loader)*args.batch_size*world,
                'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
                'upstream_commit': subprocess.check_output(['git','-C',str(ROOT/'vendor/JiT'),'rev-parse','HEAD'],text=True).strip(),
                'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'config_sha256': hashlib.sha256(Path(cli.config).read_bytes()).hexdigest(),
                'dataset_md5': manifest['md5'], 'slurm_job_id': os.environ.get('SLURM_JOB_ID')})
        writer = SummaryWriter(str(output / 'tensorboard')) if rank == 0 else None
        for epoch in range(start_epoch, args.epochs):
            sampler.set_epoch(epoch)
            ddp.train()
            total_loss = torch.zeros((), device='cuda')
            start = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            for step, (x, labels) in enumerate(loader):
                lr_sched.adjust_learning_rate(optimizer, epoch + (step // args.grad_accumulation) / steps_per_epoch, args)
                x = x.cuda(non_blocking=True).float().div_(255) * 2 - 1
                labels = labels.cuda(non_blocking=True)
                sync_now = (step + 1) % args.grad_accumulation == 0
                sync_context = nullcontext() if sync_now else ddp.no_sync()
                with sync_context:
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        loss = ddp(x, labels)
                    finite = torch.isfinite(loss).int()
                    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                    if not finite.item():
                        raise RuntimeError(f'Non-finite loss epoch={epoch+1} microstep={step}')
                    (loss / args.grad_accumulation).backward()
                if sync_now:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.synchronize()
                    update_emas(model)
                    global_step += 1
                total_loss += loss.detach()
                if sync_now and global_step % 100 == 1:
                    value = misc.all_reduce_mean(loss.item())
                    if rank == 0:
                        writer.add_scalar('train/loss', value, global_step)
                        writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step)
                        print(f'Epoch {epoch+1}/{args.epochs} step {step}/{len(loader)} loss={value:.6f}', flush=True)
            dist.all_reduce(total_loss)
            if rank == 0:
                row = {'epoch': epoch+1, 'global_step': global_step,
                       'model': args.model,
                       'loss': total_loss.item()/len(loader)/world,
                       'lr': optimizer.param_groups[0]['lr'], 'seconds': time.monotonic()-start,
                       'peak_memory_gib': torch.cuda.max_memory_allocated()/1024**3}
                append_json(output / 'train.jsonl', row)
                writer.flush()
                print(json.dumps(row), flush=True)
            if (epoch+1) % args.save_every == 0 or epoch+1 == args.epochs:
                save_checkpoint(model, optimizer, args, epoch, global_step, generator)
            if (epoch+1) % 40 == 0 and epoch+1 < args.epochs:
                evaluate(model, args, 0.9996, args.cfg, 8000, 'monitor', epoch+1)
        if writer:
            writer.close()
    else:
        assert start_epoch == args.epochs, 'Final evaluation requires the completed 200-epoch checkpoint'
    # Hold each model's official CFG fixed; select EMA on 8K then report FID-50K.
    ema_results = []
    for decay in (0.9996, 0.9998, 0.9999):
        ema_results.append(evaluate(model, args, decay, args.cfg, 8000, 'ema-select', args.epochs))
    best = min(ema_results, key=lambda r: r['frechet_inception_distance'])
    finals = [evaluate(model, args, best['ema'], args.cfg, 50000, 'final', args.epochs)]
    issue56 = None
    if args.model == 'JiT-B/16':
        issue56 = evaluate(model, args, 0.9996, args.issue56_cfg, 50000, 'issue56', args.epochs)
    if rank == 0:
        atomic_json(output / 'summary.json', {'status': 'complete', 'completed_epochs': args.epochs,
                    'model': args.model, 'resolution': args.img_size, 'official_cfg': args.cfg,
                    'ema_selection_on_8k_at_fixed_official_cfg': best,
                    'final_50k_official_cfg': finals, 'issue56_diagnostic_50k': issue56})
    dist.destroy_process_group()


@torch.no_grad()
def update_emas(model):
    for decay, targets in model.emas.items():
        for target, source in zip(targets, model.parameters()):
            target.mul_(float(decay)).add_(source, alpha=1-float(decay))


if __name__ == '__main__':
    main()
