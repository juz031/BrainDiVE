"""Offline encoders, pRF pooling and final PNG adjustment for saved BrainDiVE runs.

No diffusion model or ranking-model fitting is imported by this module.
"""

from pathlib import Path
import hashlib
import json
import math
import pickle

import numpy as np
import torch
from PIL import Image

from dual_scope_contrast_utils import (
    DualScopeContrastConvergenceError, global_luma_mean_and_rms, weighted_luma_mean_and_rms,
)
from luma_rgb_clip_utils import match_luma_rgb_clip
from prf_utils import gauss_2d, get_prf_grid
from ranking_model_training_utils import LayerSize, Normalize, load_concat_metadata, validate_concat_layout


FINAL_ACCEPTANCE_DOMAIN = 'quantized_rgb_after_clipping'
UNADJUSTED_ACCEPTANCE_DOMAIN = 'original_rgb_png_pixels'


class FinalPNGValidationError(DualScopeContrastConvergenceError):
    """A solved projection whose final uint8 RGB pixels miss active targets."""

    def __init__(self, diagnostics):
        self.diagnostics = diagnostics
        failed = ', '.join(diagnostics['failed_png_statistics'])
        super().__init__(f'Final quantized RGB missed target tolerance: {failed}',
                         reason='final_png_target_mismatch')


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def build_encoder(name, checkpoint, device):
    """Load existing local weights only, using the generation runner's architectures."""
    from torchvision.models import resnet50
    from torchvision.models.feature_extraction import create_feature_extractor

    if name == 'OPEN_CLIP_RN50':
        import open_clip
        model = open_clip.create_model('RN50', pretrained=str(checkpoint), force_quick_gelu=True)
        backbone = model.visual
        nodes = {'act3': 'relu', **{f'layer{i}': f'layer{i}' for i in range(1, 5)}}
    else:
        # These are the same torchvision ResNet-50 weights used by the original
        # DINO torch.hub and ADV robustness wrappers. Both classifier heads are
        # discarded by feature extraction; no network or CUDA-only wrapper needed.
        backbone = resnet50(weights=None)
        if name == 'DINO_RN50':
            state = torch.load(checkpoint, map_location='cpu', weights_only=True)
            backbone.fc = torch.nn.Identity()
        elif name == 'ADV_RN50':
            import dill
            state = torch.load(checkpoint, map_location='cpu', pickle_module=dill,
                               weights_only=False)['model']
            state = {k.removeprefix('module.').removeprefix('model.'): v
                     for k, v in state.items() if k.removeprefix('module.').startswith('model.')}
        else:
            raise ValueError(f'Unsupported encoder: {name}')
        backbone.load_state_dict(state, strict=True)
        nodes = {'relu': 'relu', **{f'layer{i}': f'layer{i}' for i in range(1, 5)}}
    encoder = create_feature_extractor(backbone, return_nodes=nodes).eval().to(device)
    encoder.requires_grad_(False)
    return encoder


def build_kernels(layers, model, prf_id, device='cpu'):
    """Rasterize just the selected pRF, identically to get_prf_stack's slice."""
    grid, _ = get_prf_grid()
    x, y, sigma = grid[prf_id]
    sizes = getattr(LayerSize, f'{model}_layer_size')
    return {layer: torch.as_tensor(
        gauss_2d([x, y], sigma, sizes[layer][-1], aperture=1., dtype=np.float32),
        dtype=torch.float64, device=device) for layer in layers}


def preprocess(images, model):
    images = torch.nn.functional.interpolate(images.float(), (224, 224), mode='bilinear',
                                             align_corners=False, antialias=True)
    mean = torch.as_tensor(getattr(Normalize, f'{model}_MEAN'), device=images.device)
    std = torch.as_tensor(getattr(Normalize, f'{model}_STD'), device=images.device)
    return (images - mean) / std


def validate_readout(readout, count):
    for key in ('weights', 'feature_mean', 'feature_std'):
        value = readout[key]
        if value.shape != (count,) or not np.isfinite(value).all():
            raise ValueError(f'Invalid readout {key}: {value.shape}, expected {(count,)}')
    if np.any(readout['feature_std'] <= 0) or not np.isfinite(readout['intercept']).all():
        raise ValueError('Readout contains invalid standard deviations or intercept')
    if np.asarray(readout['intercept']).size != 1:
        raise ValueError('Readout intercept must be scalar')


def load_generation(folder):
    """Retain normalization arrays on CPU as memory maps, never all voxels on GPU."""
    with open(folder / 'best_weights.pkl', 'rb') as handle:
        weights = pickle.load(handle)
    return dict(weights=weights,
                means=np.load(folder / 'best_features_m.npy', mmap_mode='r'),
                stds=np.load(folder / 'best_features_s.npy', mmap_mode='r'),
                prfs=np.load(folder / 'best_prf_idx.npy', mmap_mode='r'))


def generation_readout(loaded, voxel, count):
    w = np.asarray(loaded['weights'][str(voxel)], dtype=np.float64).reshape(-1)
    result = dict(weights=w[:-1], intercept=w[-1],
                  feature_mean=np.array(loaded['means'][voxel], dtype=np.float64),
                  feature_std=np.array(loaded['stds'][voxel], dtype=np.float64))
    validate_readout(result, count)
    return result


def load_rank_readout(folder, generation, gen_meta):
    meta = json.loads((folder / 'model.json').read_text())
    expected = (generation['subject'], generation['region'], generation['voxel_id'],
                generation['split'], generation['generation_model']['name'], generation['prf_idx'])
    actual = (meta['subject'], meta['region'], meta['voxel']['id'], meta['fit']['split_id'],
              meta['prf']['generation_model_name'], meta['prf']['id'])
    if actual != expected or meta['ranking_encoder']['name'] != 'OPEN_CLIP_RN50':
        raise ValueError(f'Ranking readout identity mismatch: {actual} != {expected}')
    if not meta['prf']['shared_with_generation_model'] or meta['prf']['grid_name'] != generation['prf_grid_name']:
        raise ValueError('Ranking pRF is not shared with the generation model')
    for field, key in (('x_aperture_units', 'prf_x'), ('y_aperture_units', 'prf_y'),
                       ('sigma_aperture_units', 'prf_sigma')):
        if not np.isclose(meta['prf'][field], generation[key], rtol=0, atol=1e-7):
            raise ValueError('Ranking pRF coordinates disagree with generation metadata')
    enc = meta['ranking_encoder']
    rank_meta = enc['checkpoint_concat_metadata']
    validate_concat_layout(rank_meta, LayerSize.OPEN_CLIP_RN50_layer_size, 'Ranking')
    if enc['concatenation_order'] != rank_meta['layers'] or generation['prf_idx'] not in rank_meta['prf_ids']:
        raise ValueError('Ranking concatenation/pRF metadata mismatch')
    if generation['generation_model']['layers'] != gen_meta['layers']:
        raise ValueError('Generation layers changed since image generation')
    expected_input = dict(color_space='RGB', range=[0., 1.], resize=[224, 224],
                          resize_mode='bilinear', antialias=True,
                          mean=Normalize.OPEN_CLIP_RN50_MEAN.reshape(-1).tolist(),
                          std=Normalize.OPEN_CLIP_RN50_STD.reshape(-1).tolist())
    if enc['input'] != expected_input:
        raise ValueError('Unsupported ranking encoder preprocessing')
    path = folder / meta['artifacts']['parameters_file']
    checksum = meta['artifacts'].get('parameters_sha256')
    if not checksum or sha256_file(path) != checksum:
        raise ValueError(f'Ranking parameters checksum missing or mismatched: {path}')
    with np.load(path, allow_pickle=False) as data:
        readout = {k: data[k].copy() for k in ('weights', 'intercept', 'feature_mean', 'feature_std')}
        if not np.array_equal(data['channel_indices'], np.arange(rank_meta['num_features'])):
            raise ValueError('Unsupported ranking channel ordering')
    validate_readout(readout, rank_meta['num_features'])
    return readout, rank_meta


@torch.inference_mode()
def score_pixels(pixels, model, encoder, layers, kernels, readout, device):
    """Score uint8 HWC arrays; these exact pixels are also written to PNG."""
    images = torch.from_numpy(np.stack(pixels)).permute(0, 3, 1, 2).to(device).float() / 255.
    extracted = encoder(preprocess(images, model))
    pooled = torch.cat([torch.einsum('bchw,hw->bc', extracted[layer].double(), kernels[layer])
                        for layer in layers], dim=1)
    tensors = {k: torch.as_tensor(v, dtype=torch.float64, device=device) for k, v in readout.items()}
    values = ((pooled - tensors['feature_mean']) / tensors['feature_std']) @ tensors['weights']
    return (values + tensors['intercept'].reshape(())).cpu().numpy()


def settings_from_generation(record):
    if record['image_save_mode'] != 'unconstrained_only':
        raise ValueError('Expected saved pre-final-adjustment PNGs')
    projection = record['constraint_projection']
    if projection['method'] != 'luma_rgb_clip' or record['contrast_constraint_method'] != 'match':
        raise ValueError('This postprocessor supports the recorded luma_rgb_clip match experiment')
    lum, dual = record['luminance'], record['dual_scope_settings']
    return dict(target_global_mean=lum['target_full_image_byte'] / 255.,
                target_prf_mean=lum['target_prf_byte'] / 255.,
                target_global_rms=record['contrast_target_rms_byte'] / 255.,
                target_prf_rms=record['contrast_target_rms_byte'] / 255.,
                mean_tolerance=lum['acceptance_tolerance_byte'] / 255.,
                rms_tolerance=record['contrast_validation_tolerance_byte'] / 255.,
                match_iterations=projection['match_iterations'], max_nfev=dual['solver_max_nfev'],
                log_scale_limit=dual['parameter_limit'], shift_limit=lum['shift_limit_normalized'],
                adjustment_envelope_power=dual['adjustment_envelope_power'])


def read_pixels(path):
    with Image.open(path) as image:
        if image.format != 'PNG' or image.mode != 'RGB':
            raise ValueError(f'Expected RGB PNG: {path}')
        return np.asarray(image).copy()


@torch.inference_mode()
def png_statistics(pixels, weight):
    """Measure the exact RGB uint8 pixels that will be scored and exported."""
    quantized = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).double() / 255.
    gm, gr = global_luma_mean_and_rms(quantized)
    pm, pr = weighted_luma_mean_and_rms(quantized, weight)
    return dict(global_mean=float(gm[0] * 255), global_rms=float(gr[0] * 255),
                prf_mean=float(pm[0] * 255), prf_rms=float(pr[0] * 255))


def prepare_unadjusted_pixels(pixels, weight):
    """Preserve source pixels, with no projection or luminance/contrast rejection."""
    return pixels, dict(method='none', acceptance_domain=UNADJUSTED_ACCEPTANCE_DOMAIN,
                        final_adjustment_applied=False, final_target_validation_applied=False,
                        png_stats_byte=png_statistics(pixels, weight))


@torch.inference_mode()
def adjust_pixels(pixels, mode, weight, settings):
    options = dict(settings)
    if mode == 'full_only':
        options.update(target_prf_mean=None, target_prf_rms=None)
    elif mode != 'full_and_prf':
        raise ValueError(f'Unknown adjustment mode: {mode}')
    source = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).float() / 255.
    output, diagnostics = match_luma_rgb_clip(source, weight, **options)
    png = (output[0].permute(1, 2, 0).clamp(0, 1) * 255).round().to(torch.uint8).numpy()
    diagnostics['png_stats_byte'] = png_statistics(png, weight)
    diagnostics['final_adjustment_applied'] = True
    diagnostics['final_target_validation_applied'] = True
    # The reference projector still has its own convergence checks. Final
    # candidate acceptance is based on the exact uint8 pixels sent to scoring
    # and export, including any errors from RGB clipping and quantization.
    diagnostics['projection_acceptance_domain'] = diagnostics['acceptance_domain']
    diagnostics['projection_postclip_mismatch_policy'] = diagnostics['postclip_mismatch_policy']
    diagnostics['acceptance_domain'] = FINAL_ACCEPTANCE_DOMAIN
    diagnostics['postclip_mismatch_policy'] = 'reject_after_quantization'
    targets, tolerances, errors, failed = {}, {}, {}, []
    for name, observed in diagnostics['png_stats_byte'].items():
        target = options[f'target_{name}']
        tolerance = 255. * options['mean_tolerance' if name.endswith('mean') else 'rms_tolerance']
        targets[name] = None if target is None else 255. * target
        tolerances[name] = None if target is None else tolerance
        errors[name] = None if target is None else abs(observed - targets[name])
        if target is not None and (not math.isfinite(observed) or errors[name] > tolerance):
            failed.append(name)
    diagnostics['png_targets_byte'] = targets
    diagnostics['png_tolerances_byte'] = tolerances
    diagnostics['png_errors_byte'] = errors
    diagnostics['failed_png_statistics'] = failed
    diagnostics['final_png_validation_passed'] = not failed
    if failed:
        raise FinalPNGValidationError(diagnostics)
    return png, diagnostics
