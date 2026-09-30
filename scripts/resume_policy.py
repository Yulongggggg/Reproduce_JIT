"""Explicit, batch-preserving four-to-eight GPU checkpoint continuation."""


def validate_resume(checkpoint, config, world, allow_expand=False):
    old_world = checkpoint['world_size']
    old = checkpoint['config']
    old_batch = old_world * old['batch_size'] * old['grad_accumulation']
    new_batch = world * config['batch_size'] * config['grad_accumulation']
    if old_batch != 1024 or new_batch != 1024:
        raise ValueError('Resume requires effective batch 1024 on both sides')
    expanding = old_world != world
    if expanding and not (allow_expand and (old_world, world) == (4, 8)):
        raise ValueError(f'GPU count transition {old_world}->{world} is not authorized')
    ignored = {'data_path', 'output_dir', 'num_workers'}
    if expanding:
        ignored |= {'batch_size', 'grad_accumulation'}
    for key, value in config.items():
        if key not in ignored and old.get(key) != value:
            raise ValueError(f'Resume config mismatch: {key}')
    if len(checkpoint['rng_per_rank']) != old_world:
        raise ValueError('Checkpoint rank RNG states are incomplete')
    if expanding and config['lr_schedule'] != 'constant':
        raise ValueError('Only the registered constant-LR experiment may expand')
    return expanding


def expansion_seed(seed, completed_epochs, rank):
    """Independent reproducible RNG for ranks that did not exist before expansion."""
    return int(seed) + 1000003 * int(completed_epochs) + int(rank)
