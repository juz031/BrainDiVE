"""Utilities for selecting voxels by their R-squared rank."""

import numpy as np


def select_voxels_by_rank(
    scores,
    top_k,
    rank_range=None,
):
    """Return indices, scores, and 1-based ranks in descending score order.

    ``top_k`` preserves the original interface and selects ranks 1 through k.
    When ``rank_range`` is supplied, its inclusive 1-based start and end ranks
    take precedence. For example, ``(6, 10)`` selects five voxels.
    """
    flat_scores = np.asarray(scores).reshape(-1)
    if rank_range is None:
        start_rank, end_rank = 1, int(top_k)
        if end_rank < 1:
            raise ValueError("Top k must be at least 1.")
        end_rank = min(end_rank, flat_scores.size)
    else:
        start_rank, end_rank = (int(value) for value in rank_range)

    if start_rank < 1:
        raise ValueError("Voxel rank START must be at least 1.")
    if end_rank < start_rank:
        raise ValueError("Voxel rank END must be greater than or equal to START.")
    if flat_scores.size == 0:
        raise ValueError("Cannot select voxels because no scores are available.")
    if end_rank > flat_scores.size:
        raise ValueError(
            f"Requested voxel rank {end_rank}, but only {flat_scores.size} "
            "voxel scores are available."
        )

    ranked_indices = np.argsort(flat_scores)[::-1]
    selected_indices = ranked_indices[start_rank - 1:end_rank]
    selected_scores = flat_scores[selected_indices]
    selected_ranks = np.arange(start_rank, end_rank + 1, dtype=np.int64)
    return selected_indices, selected_scores, selected_ranks
