"""Validate real-image loading after compilation and epoch-boundary RNG restore."""
from pathlib import Path
import tempfile

import torch
from torchvision import datasets, transforms

from run import ROOT, atomic_json, center_crop_arr, training_loader, training_transform


def main():
    import torch._inductor.config

    torch.set_num_threads(1)
    torch._inductor.config.compile_threads = 1
    images = sorted((ROOT / 'data/imagenet/train/n01440764').glob('*.JPEG'))[:32]
    assert len(images) == 32
    compiled = torch.compile(lambda x: x.sin() + x)
    assert torch.isfinite(compiled(torch.arange(32, dtype=torch.float32))).all()
    with tempfile.TemporaryDirectory(prefix='jit_loader_') as temporary:
        folder = Path(temporary) / 'n01440764'
        folder.mkdir()
        for image in images:
            (folder / image.name).symlink_to(image)
        dataset = datasets.ImageFolder(temporary, transform=training_transform(256))
        original = transforms.Compose([
            transforms.Lambda(lambda img: center_crop_arr(img, 256)),
            transforms.RandomHorizontalFlip(), transforms.PILToTensor()])
        image = dataset.loader(dataset.samples[0][0])
        torch.manual_seed(99)
        before = original(image)
        torch.manual_seed(99)
        assert torch.equal(before, dataset.transform(image))

        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=4, rank=0)
        generator = torch.Generator().manual_seed(17)
        loader = training_loader(dataset, sampler, 2, 2, generator)
        assert loader.multiprocessing_context.get_start_method() == 'spawn'
        sampler.set_epoch(0)
        first = list(loader)
        saved_rng = generator.get_state().clone()
        # Recompile for another shape before constructing the second epoch's workers.
        assert torch.isfinite(compiled(torch.arange(64, dtype=torch.float32))).all()
        sampler.set_epoch(1)
        uninterrupted = list(loader)
        restored = torch.Generator()
        restored.set_state(saved_rng)
        resumed = list(training_loader(dataset, sampler, 2, 2, restored))
        assert len(first) == len(uninterrupted) == len(resumed) == 4
        for (x, y), (other_x, other_y) in zip(uninterrupted, resumed):
            assert x.shape == (2, 3, 256, 256) and x.dtype == torch.uint8
            assert torch.equal(x, other_x) and torch.equal(y, other_y)
        assert torch.equal(generator.get_state(), restored.get_state())
    result = {'status': 'passed', 'scope': 'CPU regression, real ImageNet images',
              'torch': torch.__version__, 'workers': 2, 'start_method': 'spawn',
              'after_inductor_compile': True, 'preprocessing_unchanged': True,
              'epoch_boundary_rng_restore_equal': True,
              'gpu_resume_validated': False}
    atomic_json(ROOT / 'reports/dataloader_spawn_validation.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    main()
