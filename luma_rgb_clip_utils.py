"""Luma-only constraints, original chroma restoration, then direct RGB clipping.

Whole-image projection follows the read-only luma_mean_rms_rgb_clip reference.
The dual-scope extension fits the existing spatial transform to grayscale only;
it never scales the original color residual. Acceptance precedes RGB clipping.
"""
import math

import torch

from dual_scope_contrast_utils import (
    REC709, DualScopeContrastConvergenceError, normalized_prf_weight,
    require_finite_seed_image,
)
from joint_luma_contrast_utils import match_luma_and_rms, joint_result_to_jsonable


METHODS = ('luma_rgb_clip', 'joint_solver')
DOMAIN = 'clipped_luma_before_color_restoration_and_rgb_clipping'
NAMES = ('global_mean', 'prf_mean', 'global_rms', 'prf_rms')


def validate_luma_contrast_method(method, contrast_full, contrast_prf,
                                  luminance_full, luminance_prf,
                                  contrast_method, match_iterations):
    if method not in METHODS:
        raise ValueError(f'Unknown luma_contrast_method: {method}')
    if not isinstance(match_iterations, int) or match_iterations < 1:
        raise ValueError('luma_rgb_clip_match_iterations must be a positive integer')
    stages = {'none': 0, 'final': 1, 'every_step': 2}
    if any(s not in stages for s in (contrast_full, contrast_prf, luminance_full, luminance_prf)):
        raise ValueError('Unknown constraint timing')
    if method == 'luma_rgb_clip':
        if contrast_full != luminance_full or contrast_prf != luminance_prf:
            raise ValueError('luma_rgb_clip requires matching contrast/luminance timings within each scope; '
                             'choose joint_solver for independent timings')
        if stages[contrast_prf] > stages[contrast_full]:
            raise ValueError('pRF timing requires full-image constraints at the same or earlier stage')
        if contrast_full != 'none' and contrast_method != 'match':
            raise ValueError('luma_rgb_clip requires contrast_constraint_method=match')


def method_metadata(method, full, prf, match_iterations):
    def solver(stage):
        active = lambda s: s != 'none' if stage == 'final' else s == 'every_step'
        if not active(full):
            return 'none'
        if method == 'joint_solver':
            return 'joint_rgb_solver'
        if active(prf):
            return 'coupled_luma_torch' if stage == 'every_step' else 'coupled_luma_scipy'
        return 'reference_full_image_luma_loop'
    return dict(method=method, acceptance_domain=DOMAIN if method == 'luma_rgb_clip' else 'unquantized_float',
                postclip_mismatch_policy='record_only' if method == 'luma_rgb_clip' else 'reject',
                match_iterations=match_iterations, maximum_full_image_passes=match_iterations + 1,
                effective_solver={s: solver(s) for s in ('every_step', 'final')})


def _luma_stats(y, weight):
    mean = y.mean((-2, -1)).flatten(1)[:, 0]
    rms = (y - mean[:, None, None, None]).square().mean((1, 2, 3)).clamp_min(1e-16).sqrt()
    if weight is None:
        return (mean, None, rms, None)
    pm = (y * weight).sum((1, 2, 3))
    pr = ((y - pm[:, None, None, None]).square() * weight).sum((1, 2, 3)).clamp_min(1e-16).sqrt()
    return (mean, pm, rms, pr)


def match_luma_rgb_clip(images, spatial_weight=None, *, target_global_mean=None,
                        target_prf_mean=None, target_global_rms=None, target_prf_rms=None,
                        differentiable=False, match_iterations=20, mean_tolerance=.01/255,
                        rms_tolerance=.01/255, **solver_options):
    """Return RGB plus JSON diagnostics; retryable failures concern pre-RGB luma.

All targets/tolerances are normalized [0,1]. Full-only follows the reference's
match_iterations+1 loop. Dual scope uses a coupled solver on three identical
luma channels, reusing numerical machinery but not the old RGB transform.
"""
    if images.ndim != 4 or images.shape[1] != 3 or not images.is_floating_point():
        raise ValueError('Expected floating-point BCHW RGB images')
    for target, upper in ((target_global_mean, 1), (target_global_rms, .5)):
        if target is None or not math.isfinite(target) or not 0 <= target <= upper:
            raise ValueError('luma_rgb_clip requires valid full-image mean and RMS targets')
    if target_global_rms <= 0:
        raise ValueError('RMS target must be positive')
    if (target_prf_mean is None) != (target_prf_rms is None):
        raise ValueError('pRF mean and RMS must be enabled together')
    for tol in (mean_tolerance, rms_tolerance):
        if not math.isfinite(tol) or tol <= 0:
            raise ValueError('Tolerances must be finite and positive')
    if not isinstance(match_iterations, int) or match_iterations < 1:
        raise ValueError('match_iterations must be a positive integer')
    require_finite_seed_image(images, stage='luma_rgb_clip input')
    source = images.float().clamp(0, 1)
    coeff = source.new_tensor(REC709).view(1, 3, 1, 1)
    y = (source * coeff).sum(1, keepdim=True)
    chroma = source - y
    targets = (target_global_mean, target_prf_mean, target_global_rms, target_prf_rms)
    coupled = target_prf_mean is not None
    if coupled and spatial_weight is None:
        raise ValueError('pRF constraints require spatial_weight')
    weight = normalized_prf_weight(spatial_weight, y) if spatial_weight is not None else None
    solver_diagnostics = None
    if coupled:
        try:
            result = match_luma_and_rms(
                y.expand(-1, 3, -1, -1), spatial_weight,
                target_global_mean=target_global_mean, target_prf_mean=target_prf_mean,
                target_global_rms=target_global_rms, target_prf_rms=target_prf_rms,
                differentiable=differentiable, mean_tolerance=mean_tolerance,
                rms_tolerance=rms_tolerance, **solver_options)
        except DualScopeContrastConvergenceError as error:
            raise DualScopeContrastConvergenceError(
                f'Coupled luma projection failed: {error}',
                reason='luma_rgb_clip_solver_failure', result=error.result) from error
        z = result.images[:, :1]
        solver_diagnostics = joint_result_to_jsonable(result)
        passes = int(result.solver_iterations.max().detach().cpu())
        backend = 'coupled_luma_torch' if differentiable else 'coupled_luma_scipy'
    else:
        z = y
        for passes in range(1, match_iterations + 2):
            mean, _, rms, _ = _luma_stats(z, None)
            z = ((z - mean[:, None, None, None]) *
                 (target_global_rms / rms.clamp_min(1e-8))[:, None, None, None]
                 + target_global_mean).clamp(0, 1)
            mean, _, rms, _ = _luma_stats(z, None)
            if bool(((mean - target_global_mean).abs() <= mean_tolerance).all() &
                    ((rms - target_global_rms).abs() <= rms_tolerance).all()):
                break
        backend = 'reference_full_image_luma_loop'
    require_finite_seed_image(z, stage='luma_rgb_clip projected luma')
    pre = _luma_stats(z, weight)
    tolerances = (mean_tolerance, mean_tolerance, rms_tolerance, rms_tolerance)
    if any(t is not None and not bool(((v - t).abs() <= tol).all())
           for v, t, tol in zip(pre, targets, tolerances)):
        raise DualScopeContrastConvergenceError(
            'Luma projection missed pre-RGB-clipping tolerance',
            reason='luma_rgb_clip_solver_failure')
    unbounded = z + chroma  # alpha=1; the original color residual is never fitted.
    output = unbounded.clamp(0, 1)
    require_finite_seed_image(output, stage='luma_rgb_clip RGB output')
    with torch.no_grad():
        post = _luma_stats((output * coeff).sum(1, keepdim=True), weight)
        to_values = lambda v: None if v is None else (v.detach().flatten().cpu() * 255).tolist()
        errors = lambda stats: {name: to_values(abs(v-t)) if t is not None else None
                               for name, v, t in zip(NAMES, stats, targets)}
        clipped = (unbounded < 0) | (unbounded > 1)
        delta = output - unbounded
        diagnostics = dict(
            method='luma_rgb_clip', backend=backend, acceptance_domain=DOMAIN,
            postclip_mismatch_policy='record_only', chroma_alpha=1,
            targets_byte={n: None if t is None else t*255 for n,t in zip(NAMES,targets)},
            mean_tolerance_byte=mean_tolerance*255, rms_tolerance_byte=rms_tolerance*255,
            preclip_stats_byte={n: to_values(v) for n,v in zip(NAMES,pre)},
            postclip_stats_byte={n: to_values(v) for n,v in zip(NAMES,post)},
            preclip_errors_byte=errors(pre), postclip_errors_byte=errors(post),
            iterations_or_evaluations=passes, scalar_solver=solver_diagnostics,
            out_of_gamut_channel_fraction=clipped.float().mean((1,2,3)).cpu().tolist(),
            out_of_gamut_pixel_fraction=clipped.any(1).float().mean((1,2)).cpu().tolist(),
            clip_mae_byte=to_values(delta.abs().mean((1,2,3))),
            clip_rmse_byte=to_values(delta.square().mean((1,2,3)).sqrt()),
            clip_maximum_byte=to_values(delta.abs().flatten(1).amax(1)))
    return output, diagnostics


def accumulate_guidance_diagnostics(summary, diagnostics):
    """Compact per-seed summary rather than storing every step's large payload."""
    if summary is None:
        return
    summary['evaluations'] = summary.get('evaluations', 0) + 1
    summary['last'] = diagnostics
    for field in ('preclip_errors_byte', 'postclip_errors_byte'):
        maxima = summary.setdefault('maximum_' + field, {})
        for name, values in diagnostics[field].items():
            if values is not None:
                maxima[name] = max(maxima.get(name, 0), max(values))
