"""Batched CUDA final adjustment with per-image failure isolation.

The dual transform and luma objective match the CPU reference. A bounded,
analytic-Jacobian Levenberg-Marquardt solver replaces SciPy; solutions need not
be pixel-identical. RGB restoration, clipping, quantization and all acceptance
statistics run on CUDA before returning exact uint8 pixels to the scorers.
"""
import math
import numpy as np
import torch

from dual_scope_contrast_utils import REC709, DualScopeContrastConvergenceError, normalized_prf_weight
from postprocess_ranked_utils import FINAL_ACCEPTANCE_DOMAIN, FinalPNGValidationError

NAMES = ('global_mean', 'prf_mean', 'global_rms', 'prf_rms')
BACKEND = 'torch_cuda_batched_luma_lm_v1'


def luma_stats(y, weight):
    """B x pixels luminance, one normalized spatial weight shared by the batch."""
    mg = y.mean(1)
    mp = (y * weight).sum(1)
    rg = (y - mg[:, None]).square().mean(1).clamp_min(1e-16).sqrt()
    rp = ((y - mp[:, None]).square() * weight).sum(1).clamp_min(1e-16).sqrt()
    return torch.stack((mg, mp, rg, rp), 1)


def coupled_eval(y, weight, envelope, raw_stats, theta, targets, jacobian=True):
    """Same fixed-input scalar-luma transform as the reference, with exact Jacobian."""
    beta, gamma, delta, eta = theta.unbind(1)
    a = envelope * (y - raw_stats[:, 1, None])
    local = y + beta.expm1()[:, None] * a + delta[:, None] * envelope
    mean = local.mean(1, keepdim=True)
    scale = gamma.exp()[:, None]
    unbounded = mean + eta[:, None] + scale * (local - mean)
    z = unbounded.clamp(0, 1)
    stats = luma_stats(z, weight)
    mean_scale = targets[2:].mean().clamp_min(1 / 255.)
    residual = torch.cat(((stats[:, :2] - targets[:2]) / mean_scale,
                          (stats[:, 2:] / targets[2:]).log()), 1)
    if not jacobian:
        return z, stats, residual
    ma = a.mean(1, keepdim=True)
    me = envelope.mean()
    derivatives = torch.stack((
        beta.exp()[:, None] * (ma + scale * (a - ma)),
        scale * (local - mean),
        (me + scale * (envelope - me)).expand_as(y),
        torch.ones_like(y),
    ), 1)
    derivatives *= ((unbounded > 0) & (unbounded < 1))[:, None]
    # Differentiate means and log RMS directly. Centering terms cancel because
    # global and pRF-weighted centered luminance each have zero weighted mean.
    gradients = torch.stack((
        torch.ones_like(y) / (y.shape[1] * mean_scale),
        weight.expand_as(y) / mean_scale,
        (z - stats[:, 0, None]) / (y.shape[1] * stats[:, 2, None].square()),
        weight * (z - stats[:, 1, None]) / stats[:, 3, None].square(),
    ), 1)
    jac = torch.bmm(gradients, derivatives.transpose(1, 2))
    return z, stats, residual, jac


@torch.inference_mode()
def solve_coupled(y, weight, settings, max_iterations):
    """Independent bounded LM problems; converged/failed rows leave the active batch."""
    if max_iterations < 1:
        raise ValueError('max_iterations must be positive')
    targets = y.new_tensor([settings['target_' + n] for n in NAMES])
    envelope = (weight / weight.max().clamp_min(1e-15)).pow(settings.get('adjustment_envelope_power', .5))
    raw_stats = luma_stats(y, weight)
    theta = torch.stack((
        (raw_stats[:, 2] / raw_stats[:, 3]).clamp(1e-4, 1e4).log(),
        (targets[2] / raw_stats[:, 2]).clamp(1e-4, 1e4).log(),
        torch.zeros_like(raw_stats[:, 0]), targets[0] - raw_stats[:, 0]), 1)
    limits = y.new_tensor([settings.get('log_scale_limit', 8.)] * 2 + [settings.get('shift_limit', 1.)] * 2)
    theta = theta.clamp(-limits, limits)
    damping = y.new_full((len(y),), 1e-3)
    counts = torch.zeros(len(y), dtype=torch.int64, device=y.device)
    status = torch.zeros_like(counts)
    active = torch.ones(len(y), dtype=torch.bool, device=y.device)
    for iteration in range(max_iterations):
        ids = active.nonzero().flatten()
        if ids.numel() == 0:
            break
        q = theta[ids]
        _, stats, r, jac = coupled_eval(y[ids], weight, envelope, raw_stats[ids], q, targets)
        cost = .5 * r.square().sum(1)
        jt = jac.transpose(1, 2)
        gram = jt @ jac
        gradient = (jt @ r[:, :, None]).squeeze(-1)
        projected_gradient = q - (q - gradient).clamp(-limits, limits)
        stationary = (projected_gradient.abs().amax(1) < 1e-8) | (cost < 1e-14)
        # At an active bound, remove outward directions from the normal
        # equations so blocked scale parameters cannot stall the free shifts.
        blocked = ((q <= -limits + 1e-10) & (gradient > 0)) | ((q >= limits - 1e-10) & (gradient < 0))
        free = (~blocked).to(y.dtype)
        gram = gram * free[:, :, None] * free[:, None, :]
        gradient = gradient * free
        diagonal = gram.diagonal(dim1=1, dim2=2).clamp_min(1e-10)
        matrix = gram + torch.diag_embed(damping[ids, None] * diagonal + blocked.to(y.dtype))
        solved = torch.linalg.solve_ex(matrix, gradient[:, :, None], check_errors=False)
        step = solved.result.squeeze(-1)
        numerical_failure = (solved.info != 0) | ~torch.isfinite(step).all(1)
        step = torch.where(numerical_failure[:, None], torch.zeros_like(step), step)
        trial = (q - step).clamp(-limits, limits)
        _, _, trial_r = coupled_eval(y[ids], weight, envelope, raw_stats[ids], trial, targets, False)
        trial_cost = .5 * trial_r.square().sum(1)
        accepted = (trial_cost < cost) & ~numerical_failure & ~stationary
        theta[ids] = torch.where(accepted[:, None], trial, q)
        damping[ids] = torch.where(accepted, damping[ids] / 3., damping[ids] * 10.).clamp(1e-12, 1e12)
        counts[ids] += 1
        # Relative cost/step termination reflects a stationary fit, which must
        # still satisfy every requested pre-RGB target below.
        small_change = accepted & ((cost - trial_cost) <= 1e-8 * cost.clamp_min(1e-20))
        small_step = accepted & ((trial - q).abs().amax(1) < 1e-9)
        done = stationary | small_change | small_step
        failed = numerical_failure | ((damping[ids] >= 1e12) & ~done)
        status[ids] = torch.where(done, torch.ones_like(ids), torch.where(failed, -torch.ones_like(ids), status[ids]))
        active[ids] = ~(done | failed)
    z, stats, residual = coupled_eval(y, weight, envelope, raw_stats, theta, targets, False)
    tol = y.new_tensor([settings['mean_tolerance']] * 2 + [settings['rms_tolerance']] * 2)
    success = (status == 1) & ((stats - targets).abs() <= tol).all(1) & torch.isfinite(theta).all(1)
    return z, success, counts, status, .5 * residual.square().sum(1)


@torch.inference_mode()
def adjust_pixels_cuda(pixels, mode, weight, settings, *, max_iterations=100):
    """Return one (uint8 RGB, diagnostics) or numerical exception per source image."""
    if mode not in ('full_only', 'full_and_prf'):
        raise ValueError('Unknown adjusted mode: ' + mode)
    if not pixels:
        return []
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA adjustment requested but CUDA is unavailable')
    # Preserve the reference float32 RGB decode/luma extraction, then solve
    # the scalar luma system in float64 entirely on the GPU.
    rgb = torch.from_numpy(np.stack(pixels)).to('cuda').permute(0, 3, 1, 2).float() / 255.
    coeff = rgb.new_tensor(REC709)[None, :, None, None]
    raw_luma = (rgb * coeff).sum(1, keepdim=True)
    y = raw_luma.flatten(1).double()
    w = normalized_prf_weight(weight, rgb.double())[0, 0].flatten()
    targets = y.new_tensor([settings['target_' + n] for n in NAMES])
    active_targets = [0, 2] if mode == 'full_only' else [0, 1, 2, 3]
    tolerances = y.new_tensor([settings['mean_tolerance']] * 2 + [settings['rms_tolerance']] * 2)
    if mode == 'full_only':
        # Match the original float32 loop, freezing each image at its own first
        # successful pass so batching cannot change its selected projection.
        z = y.float()
        success = torch.zeros(len(y), dtype=torch.bool, device=y.device)
        counts = torch.zeros(len(y), dtype=torch.int64, device=y.device)
        for _ in range(settings.get('match_iterations', 50) + 1):
            mean = z.mean(1)
            rms = (z - mean[:, None]).square().mean(1).clamp_min(1e-16).sqrt()
            trial = ((z - mean[:, None]) * (settings['target_global_rms'] / rms.clamp_min(1e-8))[:, None]
                     + settings['target_global_mean']).clamp(0, 1)
            z = torch.where(success[:, None], z, trial)
            counts += (~success).long()
            mean = z.mean(1)
            rms = (z - mean[:, None]).square().mean(1).clamp_min(1e-16).sqrt()
            success |= ((mean - targets[0]).abs() <= tolerances[0]) & ((rms - targets[2]).abs() <= tolerances[2])
            if bool(success.all()):
                break
        status = success.long()
        cost = torch.zeros(len(y), dtype=torch.float64, device=y.device)
    else:
        z, success, counts, status, cost = solve_coupled(y, w, settings, max_iterations)
    z = z.float().reshape_as(raw_luma)
    pre_stats = luma_stats(z.flatten(1).double(), w)
    success &= ((pre_stats[:, active_targets] - targets[active_targets]).abs() <= tolerances[active_targets]).all(1)
    unbounded = z + (rgb - raw_luma)
    output = unbounded.clamp(0, 1)
    png_tensor = (output * 255.).round().to(torch.uint8)
    # Acceptance uses the exact final quantized RGB pixels and float64 Rec.709.
    quantized = png_tensor.double() / 255.
    qluma = (quantized * quantized.new_tensor(REC709)[None, :, None, None]).sum(1).flatten(1)
    final_stats = luma_stats(qluma, w)
    post_stats = luma_stats((output.double() * quantized.new_tensor(REC709)[None, :, None, None]).sum(1).flatten(1), w)
    del quantized
    arrays = png_tensor.permute(0, 2, 3, 1).cpu().numpy()
    final_values = (final_stats * 255).cpu().tolist()
    pre_values = (pre_stats * 255).cpu().tolist()
    post_values = (post_stats * 255).cpu().tolist()
    success_values, count_values, status_values, cost_values = [v.cpu().tolist() for v in (success, counts, status, cost)]
    outcomes = []
    for i in range(len(pixels)):
        diag = dict(method='luma_rgb_clip', backend=BACKEND, adjustment_device='cuda', solver_dtype='float64',
                    chroma_alpha=1,
                    solver_algorithm='reference_full_image_luma_loop' if mode == 'full_only' else 'bounded_analytic_lm',
                    acceptance_domain=FINAL_ACCEPTANCE_DOMAIN,
                    postclip_mismatch_policy='reject_after_quantization',
                    projection_acceptance_domain='clipped_luma_before_color_restoration_and_rgb_clipping',
                    projection_postclip_mismatch_policy='record_only',
                    final_adjustment_applied=True, final_target_validation_applied=True,
                    iterations_or_evaluations=count_values[i], solver_status=status_values[i],
                    solver_cost=cost_values[i] if math.isfinite(cost_values[i]) else None,
                    solver_converged=success_values[i],
                    gpu_solver_max_iterations=max_iterations if mode == 'full_and_prf' else None,
                    png_stats_byte=dict(zip(NAMES, final_values[i])),
                    preclip_stats_byte={n:[v] for n,v in zip(NAMES,pre_values[i])},
                    postclip_stats_byte={n:[v] for n,v in zip(NAMES,post_values[i])})
        failed, target_dict, tolerance_dict, errors = [], {}, {}, {}
        for j, name in enumerate(NAMES):
            enabled = j in active_targets
            target_dict[name] = settings['target_' + name] * 255 if enabled else None
            tolerance_dict[name] = settings['mean_tolerance' if 'mean' in name else 'rms_tolerance'] * 255 if enabled else None
            errors[name] = abs(final_values[i][j] - target_dict[name]) if enabled else None
            if enabled and (not math.isfinite(final_values[i][j]) or errors[name] > tolerance_dict[name]):
                failed.append(name)
        diag.update(png_targets_byte=target_dict, png_tolerances_byte=tolerance_dict,
                    png_errors_byte=errors, failed_png_statistics=failed,
                    final_png_validation_passed=not failed)
        if not success_values[i]:
            error = DualScopeContrastConvergenceError('GPU luma solve failed convergence or target checks', reason='luma_rgb_clip_solver_failure')
            error.diagnostics = diag
            outcomes.append(error)
        elif failed:
            outcomes.append(FinalPNGValidationError(diag))
        else:
            outcomes.append((arrays[i], diag))
    return outcomes
