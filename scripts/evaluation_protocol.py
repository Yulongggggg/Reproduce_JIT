"""Paper-aligned evaluation and strict, epoch-matched FID comparison (no GPU imports)."""
from functools import lru_cache
import hashlib
import math
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PAPER_URL = 'https://arxiv.org/html/2511.13720v2'
PAPER_PROTOCOL = 'jit-paper-v2-ema-cfg-search-8k-final-50k'
EMA_CANDIDATES = (0.9996, 0.9998, 0.9999)
CFG_CANDIDATES = tuple(i / 10 for i in range(10, 41))
PAPER_FID_200 = {'JiT-B/16': 4.37, 'JiT-L/16': 2.79, 'JiT-H/16': 2.29,
                 'JiT-B/32': 4.64, 'JiT-L/32': 3.06, 'JiT-H/32': 2.51}


@lru_cache(maxsize=None)
def metric_protocol(resolution, method, steps, interval_min, interval_max, class_num):
    stats = ROOT / 'vendor/JiT/fid_stats' / f'jit_in{resolution}_stats.npz'
    return {
        'sampling_method': method, 'num_sampling_steps': steps,
        'cfg_interval': [interval_min, interval_max],
        'reference_dataset': 'ImageNet-1K training set',
        'class_num': class_num, 'class_balanced': True,
        'fid_statistics_file': stats.name,
        'fid_statistics_sha256': hashlib.sha256(stats.read_bytes()).hexdigest(),
        'metric_backend': 'LTH14/torch-fidelity',
        'metric_backend_commit': subprocess.check_output(
            ['git', '-C', str(ROOT / 'vendor/torch-fidelity'), 'rev-parse', 'HEAD'],
            text=True).strip(),
        'pixel_conversion': 'round(clip((x+1)/2*255,0,255)), uint8 RGB PNG',
        'generation_dtype': 'bfloat16 autocast',
    }


def protocol_for(config):
    return metric_protocol(config['img_size'], config['sampling_method'],
                           config['num_sampling_steps'], config['interval_min'],
                           config['interval_max'], config['class_num'])


def validate_metric(result, config, count=50000):
    """Reject wrong-epoch, small-sample, mismatched or untraceable formal results."""
    expected = {'model': config['model'], 'resolution': config['img_size'],
                'completed_epochs': config['epochs'], 'num_images': count,
                'samples_per_class': count // config['class_num'],
                'evaluation_protocol': protocol_for(config)}
    for key, value in expected.items():
        if result.get(key) != value:
            raise ValueError(f'{key}: expected {value!r}, got {result.get(key)!r}')
    fid = result.get('frechet_inception_distance')
    if not isinstance(fid, (int, float)) or not math.isfinite(fid) or fid < 0:
        raise ValueError(f'Invalid FID: {fid!r}')
    if result.get('ema') not in EMA_CANDIDATES or result.get('cfg') not in CFG_CANDIDATES:
        raise ValueError('EMA/CFG is outside the paper candidate set')


def comparison(summary, config):
    """Only calculate a paper delta after verifying the complete selection evidence."""
    reference_resolution = 512 if config['model'].endswith('/32') else 256
    row = {'model': config['model'], 'resolution': config['img_size'],
           'completed_epochs': config['epochs'], 'num_images': 50000,
           'paper_fid': PAPER_FID_200.get(config['model'])
                if config['epochs'] == 200 and config['img_size'] == reference_resolution else None,
           'paper_source': PAPER_URL + '#S5.T6', 'our_fid': None, 'delta': None,
           'status': 'pending', 'reason': 'Waiting for completed paper-protocol FID-50K'}
    if not summary or not summary.get('final_50k_paper_protocol'):
        return row
    try:
        if summary.get('evaluation_policy') != PAPER_PROTOCOL:
            raise ValueError('Missing paper EMA/CFG selection policy')
        selected = summary['paper_selection_on_8k']
        candidates = selected['candidates']
        pairs = {(r['ema'], r['cfg']) for r in candidates}
        if len(candidates) != 93 or pairs != {(e, c) for e in EMA_CANDIDATES for c in CFG_CANDIDATES}:
            raise ValueError('Incomplete 93-point EMA/CFG selection')
        for candidate in candidates:
            validate_metric(candidate, config, count=8000)
        best = min(candidates, key=lambda r: r['frechet_inception_distance'])
        result = summary['final_50k_paper_protocol']
        validate_metric(result, config)
        if (result['ema'], result['cfg']) != (best['ema'], best['cfg']):
            raise ValueError('Final EMA/CFG does not match the minimum 8K selection FID')
        if row['paper_fid'] is None:
            raise ValueError('No published reference for this model/resolution/epoch')
        protocol = result['evaluation_protocol']
        if (protocol['sampling_method'], protocol['num_sampling_steps'],
                protocol['cfg_interval'], protocol['class_num']) != ('heun', 50, [0.1, 1.0], 1000):
            raise ValueError('Sampler/class settings differ from the paper')
        row.update(status='matched', reason=None, our_fid=result['frechet_inception_distance'],
                   delta=result['frechet_inception_distance'] - row['paper_fid'],
                   ema=result['ema'], cfg=result['cfg'], evaluation_protocol=protocol)
    except (ValueError, KeyError, TypeError) as error:
        row.update(status='ineligible', reason=str(error))
    return row


def run_final_evaluations(config, evaluate, save_summary, save_selection):
    """Durable per-point metrics let a 12-hour allocation resume the full sweep."""
    epoch = config['epochs']
    fixed = [evaluate(e, config['cfg'], 8000, 'ema-select', epoch) for e in EMA_CANDIDATES]
    best_fixed = min(fixed, key=lambda r: r['frechet_inception_distance'])
    fixed_final = evaluate(best_fixed['ema'], config['cfg'], 50000, 'final', epoch)
    validate_metric(fixed_final, config)
    summary = {'status': 'evaluating', 'completed_epochs': epoch,
               'model': config['model'], 'resolution': config['img_size'],
               'official_cfg': config['cfg'], 'evaluation_policy': PAPER_PROTOCOL,
               'ema_selection_on_8k_at_fixed_official_cfg': best_fixed,
               'final_50k_official_cfg': [fixed_final], 'issue56_diagnostic_50k': None,
               'final_50k_paper_protocol': None}
    save_summary(summary)
    if config['model'] == 'JiT-B/16':
        summary['issue56_diagnostic_50k'] = evaluate(
            0.9996, config['issue56_cfg'], 50000, 'final', epoch)
        validate_metric(summary['issue56_diagnostic_50k'], config)
        save_summary(summary)
    candidates = []
    for decay in EMA_CANDIDATES:
        for cfg in CFG_CANDIDATES:
            result = evaluate(decay, cfg, 8000, 'ema-select', epoch)
            validate_metric(result, config, count=8000)
            candidates.append(result)
            save_selection({'policy': PAPER_PROTOCOL, 'completed_epochs': epoch,
                            'completed_candidates': len(candidates), 'total_candidates': 93,
                            'candidates': candidates})
    best = min(candidates, key=lambda r: r['frechet_inception_distance'])
    summary['paper_selection_on_8k'] = {'best': best, 'candidates': candidates}
    summary['final_50k_paper_protocol'] = evaluate(best['ema'], best['cfg'], 50000, 'final', epoch)
    checked = comparison(summary, config)
    if checked['status'] != 'matched':
        raise ValueError(f'Final comparison validation failed: {checked}')
    summary['status'] = 'complete'
    summary['paper_comparison'] = checked
    save_summary(summary)
    return summary
