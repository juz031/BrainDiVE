"""Joint Rec.709 mean-luma/RMS projection with selectable active targets.

The final four-target solve follows the read-only reference in
dual_scope_luma_rms_constraints/joint_luma_contrast_utils.py. The PyTorch
solver extends that transform to differentiable brain guidance. Inactive
targets have no residual and their corresponding parameter is fixed at zero.
Every trial transforms the same fixed input, then clips before measurement.
"""

from dataclasses import dataclass
import math

import numpy as np
import torch

from dual_scope_contrast_utils import (
    REC709, DualScopeContrastConvergenceError, DualScopeContrastMatchResult,
    normalized_prf_weight, require_finite_seed_image, result_to_jsonable,
)


STAGES = {"none": 0, "final": 1, "every_step": 2}
STAT_NAMES = ("global_mean", "prf_mean", "global_rms", "prf_rms")
# Parameters are [beta, gamma, local_shift, global_shift].
PARAM_FOR_STAT = (3, 2, 1, 0)


def validate_luminance_settings(
    full_image, prf, target_full_image, target_prf, tolerance,
    contrast_full_image, contrast_method, shift_limit=1.0,
):
    for stage in (full_image, prf):
        if stage not in STAGES:
            raise ValueError(f"Unknown luminance stage: {stage!r}")
    if STAGES[prf] > STAGES[full_image]:
        raise ValueError("Luminance pRF timing requires full-image luminance "
                         "at the same or earlier stage")
    for name, stage, target in (
        ("full_image", full_image, target_full_image),
        ("prf", prf, target_prf),
    ):
        if stage != "none" and (
            target is None or not math.isfinite(target) or not 0 <= target <= 1
        ):
            raise ValueError(f"Enabled luminance {name} requires a finite "
                             "target in [0,1] (CLI units: [0,255])")
    for name, value in (("luminance tolerance", tolerance),
                        ("luminance shift limit", shift_limit)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if (full_image != "none" and contrast_full_image != "none"
            and contrast_method != "match"):
        raise ValueError("Combining luminance and contrast requires "
                         "contrast_constraint_method=match")


def active_joint_targets(
    stage, contrast_full_image, contrast_prf, luminance_full_image,
    luminance_prf, target_rms, target_mean, target_prf_mean,
):
    """Select only every_step targets in guidance; all enabled ones at final."""
    if stage not in ("every_step", "final"):
        raise ValueError(f"Unknown projection stage: {stage!r}")
    def active(setting):
        return setting != "none" if stage == "final" else setting == "every_step"
    return {
        "target_global_mean": target_mean if active(luminance_full_image) else None,
        "target_prf_mean": target_prf_mean if active(luminance_prf) else None,
        "target_global_rms": target_rms if active(contrast_full_image) else None,
        "target_prf_rms": target_rms if active(contrast_prf) else None,
    }


@dataclass(frozen=True)
class JointLumaContrastMatchResult(DualScopeContrastMatchResult):
    local_luma_shift: torch.Tensor
    global_luma_shift: torch.Tensor
    targets: tuple
    mean_tolerance: float
    rms_tolerance: float


def _stats_torch(images, weight):
    y = (images * images.new_tensor(REC709)[None, :, None, None]).sum(1)
    mg = y.mean((-2, -1))
    mp = (y * weight).sum((-2, -1))
    cg = ((y - mg[:, None, None]).square().mean((-2, -1))).clamp_min(1e-16).sqrt()
    cp = (weight * (y - mp[:, None, None]).square()).sum((-2, -1)).clamp_min(1e-16).sqrt()
    return torch.stack((mg, mp, cg, cp), dim=1)


def match_luma_and_rms(
    images, spatial_weight=None, *, target_global_mean=None, target_prf_mean=None,
    target_global_rms=None, target_prf_rms=None, differentiable=False,
    max_nfev=300, iterations=10, solver_tolerance=1e-8,
    mean_tolerance=0.01 / 255, rms_tolerance=0.01 / 255,
    log_scale_limit=8.0, shift_limit=1.0, adjustment_envelope_power=0.5,
):
    """Fit the active means/RMS jointly; classify candidate numerical failures.

    With no RMS target, the mean residual scale is 0.1. Otherwise it is the
    mean active RMS target, floored at 1/255, as in the four-target reference.
    All tolerances/targets/offsets use [0,1] pixel units.
    """
    targets = (target_global_mean, target_prf_mean, target_global_rms, target_prf_rms)
    active = [i for i, value in enumerate(targets) if value is not None]
    if not active:
        raise ValueError("At least one joint target must be enabled")
    if images.ndim != 4 or images.shape[1] != 3 or not images.is_floating_point():
        raise ValueError("Expected floating-point BCHW RGB images")
    for i in active:
        value = targets[i]
        if not math.isfinite(value) or not (
            0 <= value <= 1 if i < 2 else 0 < value <= 0.5
        ):
            raise ValueError(f"Invalid target for {STAT_NAMES[i]}: {value}")
    for value in (mean_tolerance, rms_tolerance, log_scale_limit, shift_limit,
                  adjustment_envelope_power, solver_tolerance, max_nfev, iterations):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Joint solver controls must be finite and positive")
    if spatial_weight is None:
        if target_prf_mean is not None or target_prf_rms is not None:
            raise ValueError("pRF targets require a pRF weight")
        spatial_weight = images.new_ones(images.shape[-2:])
    require_finite_seed_image(images, stage="joint projection input")
    if not differentiable and (torch.any(images < 0) or torch.any(images > 1)):
        raise DualScopeContrastConvergenceError(
            "Final joint input pixels outside [0,1]", reason="out_of_range_pixels")
    values = images.float().clamp(0, 1) if differentiable else images.detach().cpu().double()
    weight = normalized_prf_weight(spatial_weight, values)[0, 0]
    envelope = (weight / weight.max().clamp_min(1e-15)).pow(adjustment_envelope_power)
    raw_stats = _stats_torch(values, weight)
    active_params = sorted(PARAM_FOR_STAT[i] for i in active)
    bounds = [log_scale_limit, log_scale_limit, shift_limit, shift_limit]
    rms_targets = [v for v in targets[2:] if v is not None]
    mean_scale = max(sum(rms_targets) / len(rms_targets), 1 / 255) if rms_targets else 0.1
    tolerances = [mean_tolerance, mean_tolerance, rms_tolerance, rms_tolerance]

    if differentiable:
        # Use the same initialization as the reference when all targets exist.
        zero = values.new_zeros(values.shape[0])
        theta = torch.stack((
            torch.log((raw_stats[:, 2] / raw_stats[:, 3].clamp_min(1e-12)).clamp(1e-4, 1e4)),
            torch.log((float(target_global_rms or 1) / raw_stats[:, 2].clamp_min(1e-12)).clamp(1e-4, 1e4)),
            zero, (float(target_global_mean or 0) - raw_stats[:, 0]),
        ), dim=1)[:, active_params]
        limit = values.new_tensor(bounds)[active_params]
        theta = theta.clamp(-limit, limit)
        basis = torch.eye(4, device=values.device, dtype=values.dtype)[active_params]
        desired = values.new_tensor([targets[i] for i in active])
        scales = values.new_tensor([mean_scale if i < 2 else 1 for i in active])
        is_mean = values.new_tensor([i < 2 for i in active], dtype=torch.bool)
        tol = values.new_tensor([tolerances[i] for i in active])

        def project(q):
            p = q @ basis
            local = (values + torch.expm1(p[:, 0, None, None, None]) * envelope
                     * (values - raw_stats[:, 1, None, None, None])
                     + p[:, 2, None, None, None] * envelope)
            mean = _stats_torch(local, weight)[:, 0, None, None, None]
            return (mean + p[:, 3, None, None, None]
                    + torch.exp(p[:, 1, None, None, None]) * (local - mean)).clamp(0, 1)

        def residual(q):
            stats = _stats_torch(project(q), weight)[:, active]
            return torch.where(is_mean, (stats - desired) / scales,
                               torch.log(stats.clamp_min(1e-12) / desired.clamp_min(1e-12)))

        identity = torch.eye(len(active), device=values.device, dtype=values.dtype)[None]
        try:
            used_iterations = 0
            for iteration in range(int(iterations)):
                r = residual(theta)
                # Finite differences remain in the graph, including the fit.
                columns = []
                for k in range(len(active_params)):
                    offset = torch.zeros_like(theta)
                    offset[:, k] = 1e-3
                    columns.append((residual(theta + offset) - residual(theta - offset)) / 2e-3)
                jac = torch.stack(columns, dim=2)
                jt = jac.transpose(1, 2)
                step = torch.linalg.solve(jt @ jac + 1e-6 * identity,
                                          jt @ r[:, :, None]).squeeze(-1)
                # Backtracking keeps difficult clipped trials from taking a
                # large uphill step; selected trials remain differentiable.
                best = theta
                best_cost = r.square().sum(1)
                for factor in (1.0, 0.5, 0.25, 0.125, 0.0625):
                    trial = (theta - factor * step).clamp(-limit, limit)
                    cost = residual(trial).square().sum(1)
                    better = cost < best_cost
                    best = torch.where(better[:, None], trial, best)
                    best_cost = torch.where(better, cost, best_cost)
                theta = best
                used_iterations = iteration + 1
                if bool(((_stats_torch(project(theta), weight)[:, active] - desired).abs() <= tol).all()):
                    break
        except torch.linalg.LinAlgError as error:
            raise DualScopeContrastConvergenceError(
                "Joint PyTorch linear solve failed", reason="joint_solver_failure") from error
        matched = project(theta)
        parameters = theta @ basis
        matched_stats = _stats_torch(matched, weight)
        converged = ((matched_stats[:, active] - desired).abs() <= tol).all(1)
        converged &= torch.isfinite(parameters).all(1)
        counts = torch.full_like(converged, used_iterations, dtype=torch.int64)
        status = converged.to(torch.int64)
        cost = 0.5 * residual(theta).square().sum(1)
        messages = tuple("torch_differentiable_joint" for _ in range(len(values)))
        backend = "torch_joint_gauss_newton"
    else:
        from scipy.optimize import least_squares

        raw_np, w, env = values.numpy(), weight.numpy(), envelope.numpy()
        coeff = np.asarray(REC709).reshape(3, 1, 1)
        def statistics(x):
            y = (x * coeff).sum(0)
            mg, mp = y.mean(), (w * y).sum()
            return np.array((mg, mp, np.sqrt(((y - mg)**2).mean()),
                             np.sqrt((w * (y - mp)**2).sum())))
        outputs, params, initial_stats, final_stats = [], [], [], []
        successes, evaluations, statuses, costs, messages = [], [], [], [], []
        desired = np.array([targets[i] for i in active])
        limit = np.array(bounds)[active_params]
        for raw in raw_np:
            stats = statistics(raw)
            def unpack(q):
                p = np.zeros(4)
                p[active_params] = q
                return p
            def project(q):
                beta, gamma, delta, eta = unpack(q)
                local = raw + np.expm1(beta) * env * (raw - stats[1]) + delta * env
                mean = (local * coeff).sum(0).mean()
                return np.clip(mean + eta + np.exp(gamma) * (local - mean), 0, 1)
            def residual(q):
                measured = statistics(project(q))
                return np.array([(measured[i] - targets[i]) / mean_scale if i < 2
                                 else np.log(max(measured[i], 1e-12) / targets[i]) for i in active])
            initial = np.array((
                np.log(np.clip(stats[2] / max(stats[3], 1e-12), 1e-4, 1e4)),
                np.log(np.clip((target_global_rms or 1) / max(stats[2], 1e-12), 1e-4, 1e4)),
                0, (target_global_mean or 0) - stats[0],
            ))[active_params]
            try:
                fit = least_squares(residual, np.clip(initial, -limit, limit),
                                    bounds=(-limit, limit), method="trf", x_scale="jac",
                                    ftol=solver_tolerance, xtol=solver_tolerance,
                                    gtol=solver_tolerance, max_nfev=int(max_nfev))
            except (ValueError, FloatingPointError, OverflowError, np.linalg.LinAlgError) as error:
                raise DualScopeContrastConvergenceError(
                    "Joint SciPy numerical solve failed", reason="joint_solver_failure") from error
            output = project(fit.x)
            observed = statistics(output)
            successes.append(bool(fit.success) and bool(np.all(
                np.abs(observed[active] - desired) <= np.array(tolerances)[active])))
            outputs.append(output); params.append(unpack(fit.x))
            initial_stats.append(stats); final_stats.append(observed)
            evaluations.append(fit.nfev); statuses.append(fit.status)
            costs.append(fit.cost); messages.append(str(fit.message))
        def tensor(x, dtype=images.dtype):
            return torch.as_tensor(np.asarray(x), device=images.device, dtype=dtype)
        matched = tensor(outputs)
        parameters = tensor(params)
        raw_stats, matched_stats = tensor(initial_stats), tensor(final_stats)
        converged = tensor(successes, torch.bool)
        counts, status = tensor(evaluations, torch.int64), tensor(statuses, torch.int64)
        cost = tensor(costs)
        backend = "scipy_joint_float64_least_squares"
        weight = normalized_prf_weight(spatial_weight, matched)[0, 0]

    require_finite_seed_image(matched, stage="joint projection output")
    if not bool(torch.isfinite(parameters).all()):
        raise DualScopeContrastConvergenceError(
            "Non-finite joint parameters", reason="nonfinite_joint_parameters")
    # Recheck the returned pixel precision, not only SciPy's float64 trial.
    actual = _stats_torch(matched.double(), weight.double())
    desired_tensor = actual.new_tensor([targets[i] for i in active])
    converged = converged & ((actual[:, active] - desired_tensor).abs()
                             <= actual.new_tensor([tolerances[i] for i in active])).all(1)
    clipped = ((matched <= 0.5 / 255) | (matched >= 254.5 / 255)).to(matched.dtype)
    result = JointLumaContrastMatchResult(
        images=matched, raw_global_mean=raw_stats[:, 0], raw_prf_mean=raw_stats[:, 1],
        raw_global_rms=raw_stats[:, 2], raw_prf_rms=raw_stats[:, 3],
        matched_global_mean=matched_stats[:, 0], matched_prf_mean=matched_stats[:, 1],
        matched_global_rms=matched_stats[:, 2], matched_prf_rms=matched_stats[:, 3],
        local_log_scale=parameters[:, 0], global_log_scale=parameters[:, 1],
        local_luma_shift=parameters[:, 2], global_luma_shift=parameters[:, 3],
        converged=converged, solver_iterations=counts, solver_status=status,
        solver_cost=cost, solver_messages=tuple(messages),
        adjustment_envelope_power=adjustment_envelope_power,
        global_clipped_channel_fraction=clipped.mean((1, 2, 3)),
        prf_weighted_clipped_channel_fraction=(clipped.mean(1) * weight).sum((-2, -1)),
        backend=backend, targets=targets, mean_tolerance=mean_tolerance,
        rms_tolerance=rms_tolerance,
    )
    if not bool(converged.all()):
        raise DualScopeContrastConvergenceError(
            "Joint projection missed one or more requested targets",
            reason="joint_solver_failure", result=result)
    return result


def joint_result_to_jsonable(result):
    # Existing RMS diagnostic fields remain available to downstream readers.
    common_rms = result.targets[2] if result.targets[2] is not None else 0.0
    data = result_to_jsonable(result, target_rms_byte=255 * common_rms,
                             tolerance_rms_byte=255 * result.rms_tolerance)
    data["solver_nfev"] = int(result.solver_iterations[0]) if result.backend.startswith("scipy") else None
    data["local_luma_shift"] = float(result.local_luma_shift[0].detach().cpu())
    data["global_luma_shift"] = float(result.global_luma_shift[0].detach().cpu())
    data["mean_tolerance_byte"] = 255 * result.mean_tolerance
    data["targets_byte"] = {name: 255 * value if value is not None else None
                            for name, value in zip(STAT_NAMES, result.targets)}
    fields = (result.matched_global_mean, result.matched_prf_mean,
              result.matched_global_rms, result.matched_prf_rms)
    data["errors_byte"] = {name: 255 * abs(float(field[0].detach().cpu()) - value)
                           if value is not None else None
                           for name, value, field in zip(STAT_NAMES, result.targets, fields)}
    if result.targets[2] is None:
        data["target_rms_byte"] = None
        data["global_rms_error_byte"] = None
    if result.targets[3] is None:
        data["prf_rms_error_byte"] = None
    return data
