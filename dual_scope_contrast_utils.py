"""Global plus soft-pRF Rec.709 RMS contrast matching.

The final solver is a direct adaptation of the audited reference experiment:
it uses bounded SciPy least squares in float64 and is not differentiable.  The
every-step solver uses the same two-parameter image transform and residuals in
PyTorch so brain-guidance gradients remain connected to the diffusion latent.
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


REC709 = (0.2126, 0.7152, 0.0722)


class DualScopeContrastConvergenceError(RuntimeError):
    """Raised when a strict dual-scope projection does not converge."""

    def __init__(self, message, *, reason, result=None):
        super().__init__(message)
        self.reason = reason
        self.result = result


def require_finite_seed_image(images, *, stage):
    """Classify seed-specific NaN/Inf pixels as a retryable rejection."""
    if not torch.is_tensor(images):
        raise TypeError("images must be a torch tensor")
    if not bool(torch.isfinite(images).all()):
        raise DualScopeContrastConvergenceError(
            f"Generated image contains NaN or Inf during {stage}",
            reason="nonfinite_pixels",
        )


@dataclass(frozen=True)
class DualScopeContrastMatchResult:
    images: torch.Tensor
    raw_global_rms: torch.Tensor
    raw_prf_rms: torch.Tensor
    matched_global_rms: torch.Tensor
    matched_prf_rms: torch.Tensor
    raw_global_mean: torch.Tensor
    raw_prf_mean: torch.Tensor
    matched_global_mean: torch.Tensor
    matched_prf_mean: torch.Tensor
    local_log_scale: torch.Tensor
    global_log_scale: torch.Tensor
    converged: torch.Tensor
    solver_iterations: torch.Tensor
    solver_status: torch.Tensor
    solver_cost: torch.Tensor
    solver_messages: tuple
    adjustment_envelope_power: float
    global_clipped_channel_fraction: torch.Tensor
    prf_weighted_clipped_channel_fraction: torch.Tensor
    backend: str


def normalized_prf_weight(spatial_weight, reference):
    """Resize a soft pRF to ``reference`` and normalize it to unit mass."""
    weight = torch.as_tensor(
        spatial_weight,
        dtype=reference.dtype,
        device=reference.device,
    )
    if weight.ndim == 2:
        weight = weight[None, None]
    elif weight.ndim == 3:
        weight = weight[None]
    if weight.ndim != 4 or weight.shape[0] != 1 or weight.shape[1] != 1:
        raise ValueError(
            "pRF weight must have shape HW, 1HW, or 11HW; "
            f"got {tuple(weight.shape)}"
        )
    if tuple(weight.shape[-2:]) != tuple(reference.shape[-2:]):
        weight = F.interpolate(
            weight,
            size=reference.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    if not torch.isfinite(weight).all() or torch.any(weight < 0):
        raise ValueError("pRF weight must be finite and non-negative")
    total = weight.sum()
    if not bool(total > 0):
        raise ValueError("pRF weight must have positive mass")
    return weight / total


def global_luma_mean_and_rms(images):
    """Return per-image global Rec.709 mean and population RMS."""
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(f"Expected BCHW RGB images, got {tuple(images.shape)}")
    coefficients = images.new_tensor(REC709).view(1, 3, 1, 1)
    luma = (images * coefficients).sum(dim=1, keepdim=True)
    mean = luma.mean(dim=(-2, -1)).reshape(-1)
    variance = (
        (luma - mean.view(-1, 1, 1, 1)).square()
    ).mean(dim=(-2, -1)).reshape(-1)
    return mean, torch.sqrt(variance.clamp_min(1e-16))


def weighted_luma_mean_and_rms(images, spatial_weight):
    """Return soft-pRF-weighted Rec.709 mean and population RMS."""
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(f"Expected BCHW RGB images, got {tuple(images.shape)}")
    weight = normalized_prf_weight(spatial_weight, images)
    coefficients = images.new_tensor(REC709).view(1, 3, 1, 1)
    luma = (images * coefficients).sum(dim=1, keepdim=True)
    mean = (weight * luma).sum(dim=(-2, -1)).reshape(-1)
    variance = (
        weight * (luma - mean.view(-1, 1, 1, 1)).square()
    ).sum(dim=(-2, -1)).reshape(-1)
    return mean, torch.sqrt(variance.clamp_min(1e-16))


def _validate_inputs(images, target_rms, tolerance, parameter_limit, envelope_power):
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(f"Expected BCHW RGB images, got {tuple(images.shape)}")
    if not torch.is_floating_point(images):
        raise ValueError("images must be floating point")
    if not torch.isfinite(images).all():
        raise ValueError("images must be finite")
    if not 0 < float(target_rms) <= 0.5:
        raise ValueError("target_rms must lie in (0, 0.5]")
    if tolerance <= 0 or parameter_limit <= 0 or envelope_power <= 0:
        raise ValueError(
            "tolerance, parameter_limit, and envelope_power must be positive"
        )


def _torch_project(images, weight, local_log_scale, global_log_scale, envelope_power):
    raw_prf_mean, _ = weighted_luma_mean_and_rms(images, weight)
    envelope = (
        weight / weight.max().clamp_min(1e-15)
    ).pow(float(envelope_power))
    locally_adjusted = images + torch.expm1(local_log_scale).view(
        -1, 1, 1, 1
    ) * envelope * (images - raw_prf_mean.view(-1, 1, 1, 1))
    local_global_mean, _ = global_luma_mean_and_rms(locally_adjusted)
    return (
        local_global_mean.view(-1, 1, 1, 1)
        + torch.exp(global_log_scale).view(-1, 1, 1, 1)
        * (
            locally_adjusted
            - local_global_mean.view(-1, 1, 1, 1)
        )
    ).clamp(0.0, 1.0)


def _clipped_fractions(images, weight):
    clipped = (
        (images <= (0.5 / 255.0)) | (images >= (254.5 / 255.0))
    ).to(images.dtype)
    global_clipped = clipped.mean(dim=(1, 2, 3))
    prf_clipped = (
        clipped.mean(dim=1, keepdim=True) * weight
    ).sum(dim=(-2, -1)).reshape(-1)
    return global_clipped, prf_clipped


def match_global_and_prf_rms_contrast_torch(
    images,
    spatial_weight,
    target_rms,
    *,
    iterations=10,
    tolerance=0.01 / 255.0,
    parameter_limit=8.0,
    adjustment_envelope_power=0.5,
    finite_difference_step=1e-3,
    damping=1e-6,
):
    """Differentiably match both RMS statistics for every-step guidance.

    A fixed-iteration, batched damped Gauss-Newton solve is unrolled in
    PyTorch.  The decoded input is clamped to the reference solver's [0,1]
    domain before fitting.
    """
    _validate_inputs(
        images,
        target_rms,
        tolerance,
        parameter_limit,
        adjustment_envelope_power,
    )
    if iterations <= 0 or finite_difference_step <= 0 or damping <= 0:
        raise ValueError(
            "iterations, finite_difference_step, and damping must be positive"
        )

    values = images.clamp(0.0, 1.0)
    weight = normalized_prf_weight(spatial_weight, values)
    raw_global_mean, raw_global_rms = global_luma_mean_and_rms(values)
    raw_prf_mean, raw_prf_rms = weighted_luma_mean_and_rms(values, weight)
    target = values.new_full(raw_global_rms.shape, float(target_rms))
    theta = torch.stack(
        (
            torch.log(
                (raw_global_rms / raw_prf_rms.clamp_min(1e-12)).clamp(
                    1e-4, 1e4
                )
            ),
            torch.log(
                (target / raw_global_rms.clamp_min(1e-12)).clamp(1e-4, 1e4)
            ),
        ),
        dim=1,
    ).clamp(-parameter_limit, parameter_limit)

    def project(parameters):
        return _torch_project(
            values,
            weight,
            parameters[:, 0],
            parameters[:, 1],
            adjustment_envelope_power,
        )

    def residual(parameters):
        projected = project(parameters)
        _, global_rms = global_luma_mean_and_rms(projected)
        _, prf_rms = weighted_luma_mean_and_rms(projected, weight)
        return torch.stack(
            (
                torch.log(global_rms.clamp_min(1e-12) / target),
                torch.log(prf_rms.clamp_min(1e-12) / target),
            ),
            dim=1,
        )

    step_size = float(finite_difference_step)
    identity = torch.eye(2, dtype=values.dtype, device=values.device)[None]
    for _ in range(int(iterations)):
        current_residual = residual(theta)
        columns = []
        for parameter_index in range(2):
            offset = torch.zeros_like(theta)
            offset[:, parameter_index] = step_size
            columns.append(
                (residual(theta + offset) - residual(theta - offset))
                / (2.0 * step_size)
            )
        jacobian = torch.stack(columns, dim=2)
        jacobian_t = jacobian.transpose(1, 2)
        normal_matrix = jacobian_t @ jacobian + float(damping) * identity
        normal_rhs = (jacobian_t @ current_residual[:, :, None]).squeeze(-1)
        try:
            update = torch.linalg.solve(
                normal_matrix, normal_rhs[:, :, None]
            ).squeeze(-1)
        except RuntimeError as error:
            raise DualScopeContrastConvergenceError(
                "Differentiable dual-scope solve failed",
                reason="dual_scope_solver_failure",
            ) from error
        theta = (theta - update).clamp(-parameter_limit, parameter_limit)

    matched = project(theta)
    matched_global_mean, matched_global_rms = global_luma_mean_and_rms(matched)
    matched_prf_mean, matched_prf_rms = weighted_luma_mean_and_rms(matched, weight)
    global_error = (matched_global_rms - target).abs()
    prf_error = (matched_prf_rms - target).abs()
    finite = (
        torch.isfinite(theta).all(dim=1)
        & torch.isfinite(matched).flatten(1).all(dim=1)
    )
    converged = finite & (global_error <= tolerance) & (prf_error <= tolerance)
    final_residual = torch.stack(
        (
            torch.log(matched_global_rms.clamp_min(1e-12) / target),
            torch.log(matched_prf_rms.clamp_min(1e-12) / target),
        ),
        dim=1,
    )
    global_clipped, prf_clipped = _clipped_fractions(matched, weight)
    batch_size = images.shape[0]
    return DualScopeContrastMatchResult(
        images=matched,
        raw_global_rms=raw_global_rms,
        raw_prf_rms=raw_prf_rms,
        matched_global_rms=matched_global_rms,
        matched_prf_rms=matched_prf_rms,
        raw_global_mean=raw_global_mean,
        raw_prf_mean=raw_prf_mean,
        matched_global_mean=matched_global_mean,
        matched_prf_mean=matched_prf_mean,
        local_log_scale=theta[:, 0],
        global_log_scale=theta[:, 1],
        converged=converged,
        solver_iterations=torch.full(
            (batch_size,), int(iterations), dtype=torch.int64, device=images.device
        ),
        solver_status=converged.to(torch.int64),
        solver_cost=0.5 * final_residual.square().sum(dim=1),
        solver_messages=tuple("fixed_iteration_torch" for _ in range(batch_size)),
        adjustment_envelope_power=float(adjustment_envelope_power),
        global_clipped_channel_fraction=global_clipped,
        prf_weighted_clipped_channel_fraction=prf_clipped,
        backend="torch_differentiable_gauss_newton",
    )


def match_global_and_prf_rms_contrast_final(
    images,
    spatial_weight,
    target_rms,
    *,
    max_nfev=300,
    solver_tolerance=1e-12,
    tolerance=0.01 / 255.0,
    parameter_limit=8.0,
    adjustment_envelope_power=0.5,
):
    """Reference-compatible final float64 SciPy dual-scope solve."""
    _validate_inputs(
        images,
        target_rms,
        tolerance,
        parameter_limit,
        adjustment_envelope_power,
    )
    if torch.any(images < 0) or torch.any(images > 1):
        raise ValueError("final-solver images must lie in [0,1]")
    if max_nfev <= 0 or solver_tolerance <= 0:
        raise ValueError("max_nfev and solver_tolerance must be positive")
    try:
        from scipy.optimize import least_squares
    except ImportError as error:
        raise RuntimeError("Final dual-scope matching requires scipy") from error

    original_dtype = images.dtype
    original_device = images.device
    values_np = images.detach().to(device="cpu", dtype=torch.float64).numpy()
    reference = torch.from_numpy(values_np)
    weight_np = normalized_prf_weight(spatial_weight, reference)[0, 0].numpy()
    envelope_np = (
        weight_np / max(float(weight_np.max()), 1e-15)
    ) ** float(adjustment_envelope_power)
    rec709 = np.asarray(REC709, dtype=np.float64).reshape(3, 1, 1)

    def statistics(value):
        luma = np.sum(value * rec709, axis=0)
        global_mean = float(luma.mean())
        global_rms = float(np.sqrt(np.mean((luma - global_mean) ** 2)))
        prf_mean = float(np.sum(weight_np * luma))
        prf_rms = float(np.sqrt(np.sum(weight_np * (luma - prf_mean) ** 2)))
        return global_mean, global_rms, prf_mean, prf_rms

    matched_values = []
    raw_statistics = []
    matched_statistics = []
    parameters = []
    successes = []
    nfevs = []
    statuses = []
    costs = []
    messages = []
    target_value = float(target_rms)

    for raw in values_np:
        raw_stats = statistics(raw)
        raw_global_mean, raw_global_rms, raw_prf_mean, raw_prf_rms = raw_stats

        def project(parameter):
            local_log_scale, global_log_scale = parameter
            locally_adjusted = raw + np.expm1(local_log_scale) * envelope_np * (
                raw - raw_prf_mean
            )
            local_luma = np.sum(locally_adjusted * rec709, axis=0)
            local_global_mean = float(local_luma.mean())
            return np.clip(
                local_global_mean
                + np.exp(global_log_scale)
                * (locally_adjusted - local_global_mean),
                0.0,
                1.0,
            )

        def residual(parameter):
            _, global_rms, _, prf_rms = statistics(project(parameter))
            return np.asarray(
                (
                    np.log(max(global_rms, 1e-12) / target_value),
                    np.log(max(prf_rms, 1e-12) / target_value),
                ),
                dtype=np.float64,
            )

        initial = np.asarray(
            (
                np.log(np.clip(raw_global_rms / max(raw_prf_rms, 1e-12), 1e-4, 1e4)),
                np.log(np.clip(target_value / max(raw_global_rms, 1e-12), 1e-4, 1e4)),
            ),
            dtype=np.float64,
        )
        initial = np.clip(initial, -parameter_limit, parameter_limit)
        try:
            fit = least_squares(
                residual,
                initial,
                bounds=(-parameter_limit, parameter_limit),
                ftol=solver_tolerance,
                xtol=solver_tolerance,
                gtol=solver_tolerance,
                max_nfev=max_nfev,
                method="trf",
                x_scale="jac",
            )
        except (ValueError, FloatingPointError, OverflowError) as error:
            raise DualScopeContrastConvergenceError(
                "Final dual-scope numerical solve failed for this seed",
                reason="dual_scope_solver_failure",
            ) from error
        matched_image = project(fit.x)
        final_stats = statistics(matched_image)
        finite = np.isfinite(fit.x).all() and np.isfinite(matched_image).all()
        successes.append(
            bool(fit.success)
            and bool(finite)
            and abs(final_stats[1] - target_value) <= tolerance
            and abs(final_stats[3] - target_value) <= tolerance
        )
        matched_values.append(matched_image)
        raw_statistics.append(raw_stats)
        matched_statistics.append(final_stats)
        parameters.append(np.asarray(fit.x, dtype=np.float64))
        nfevs.append(int(fit.nfev))
        statuses.append(int(fit.status))
        costs.append(float(fit.cost))
        messages.append(str(fit.message))

    matched = torch.from_numpy(np.stack(matched_values)).to(
        device=original_device, dtype=original_dtype
    )
    weight = normalized_prf_weight(spatial_weight, matched)
    global_clipped, prf_clipped = _clipped_fractions(matched, weight)
    raw_stats = np.asarray(raw_statistics, dtype=np.float64)
    final_stats = np.asarray(matched_statistics, dtype=np.float64)
    parameters = np.stack(parameters)

    def column(values, index):
        return torch.from_numpy(values[:, index]).to(
            device=original_device, dtype=original_dtype
        )

    return DualScopeContrastMatchResult(
        images=matched,
        raw_global_rms=column(raw_stats, 1),
        raw_prf_rms=column(raw_stats, 3),
        matched_global_rms=column(final_stats, 1),
        matched_prf_rms=column(final_stats, 3),
        raw_global_mean=column(raw_stats, 0),
        raw_prf_mean=column(raw_stats, 2),
        matched_global_mean=column(final_stats, 0),
        matched_prf_mean=column(final_stats, 2),
        local_log_scale=torch.from_numpy(parameters[:, 0]).to(
            device=original_device, dtype=original_dtype
        ),
        global_log_scale=torch.from_numpy(parameters[:, 1]).to(
            device=original_device, dtype=original_dtype
        ),
        converged=torch.as_tensor(successes, device=original_device, dtype=torch.bool),
        solver_iterations=torch.as_tensor(nfevs, device=original_device, dtype=torch.int64),
        solver_status=torch.as_tensor(statuses, device=original_device, dtype=torch.int64),
        solver_cost=torch.as_tensor(costs, device=original_device, dtype=original_dtype),
        solver_messages=tuple(messages),
        adjustment_envelope_power=float(adjustment_envelope_power),
        global_clipped_channel_fraction=global_clipped,
        prf_weighted_clipped_channel_fraction=prf_clipped,
        backend="scipy_float64_least_squares",
    )


def require_converged(result, *, stage):
    """Raise a classified error unless every image met both RMS thresholds."""
    parameters_finite = (
        torch.isfinite(result.local_log_scale).all()
        and torch.isfinite(result.global_log_scale).all()
    )
    if not bool(parameters_finite):
        raise DualScopeContrastConvergenceError(
            f"Non-finite dual-scope parameters during {stage}",
            reason="nonfinite_dual_scope_parameters",
            result=result,
        )
    if not bool(torch.all(result.converged)):
        raise DualScopeContrastConvergenceError(
            f"Dual-scope contrast solver did not converge during {stage}",
            reason="dual_scope_solver_failure",
            result=result,
        )


def result_to_jsonable(result, *, target_rms_byte, tolerance_rms_byte):
    """Convert a one-image result into JSON-safe audit metadata."""
    if result.images.shape[0] != 1:
        raise ValueError("JSON conversion expects a one-image result")

    def scalar(value):
        return float(value.detach().cpu().reshape(-1)[0])

    return {
        "backend": result.backend,
        "target_rms_byte": float(target_rms_byte),
        "tolerance_rms_byte": float(tolerance_rms_byte),
        "raw_global_rms_byte": 255.0 * scalar(result.raw_global_rms),
        "raw_prf_rms_byte": 255.0 * scalar(result.raw_prf_rms),
        "matched_global_rms_byte": 255.0 * scalar(result.matched_global_rms),
        "matched_prf_rms_byte": 255.0 * scalar(result.matched_prf_rms),
        "global_rms_error_byte": abs(
            255.0 * scalar(result.matched_global_rms) - float(target_rms_byte)
        ),
        "prf_rms_error_byte": abs(
            255.0 * scalar(result.matched_prf_rms) - float(target_rms_byte)
        ),
        "raw_global_luminance_mean_byte": 255.0 * scalar(result.raw_global_mean),
        "raw_prf_luminance_mean_byte": 255.0 * scalar(result.raw_prf_mean),
        "matched_global_luminance_mean_byte": 255.0 * scalar(
            result.matched_global_mean
        ),
        "matched_prf_luminance_mean_byte": 255.0 * scalar(
            result.matched_prf_mean
        ),
        "local_log_scale": scalar(result.local_log_scale),
        "global_log_scale": scalar(result.global_log_scale),
        "solver_converged": bool(result.converged.detach().cpu().reshape(-1)[0]),
        "solver_iterations": int(
            result.solver_iterations.detach().cpu().reshape(-1)[0]
        ),
        "solver_nfev": (
            int(result.solver_iterations.detach().cpu().reshape(-1)[0])
            if result.backend == "scipy_float64_least_squares"
            else None
        ),
        "solver_status": int(result.solver_status.detach().cpu().reshape(-1)[0]),
        "solver_cost": scalar(result.solver_cost),
        "solver_message": result.solver_messages[0],
        "adjustment_envelope_power": float(result.adjustment_envelope_power),
        "global_clipped_channel_fraction": scalar(
            result.global_clipped_channel_fraction
        ),
        "prf_weighted_clipped_channel_fraction": scalar(
            result.prf_weighted_clipped_channel_fraction
        ),
    }
