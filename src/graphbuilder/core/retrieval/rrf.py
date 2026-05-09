"""Reciprocal Rank Fusion (Cormack et al. 2009).

RRF is the canonical way to combine results from heterogeneous rankers
when the underlying scores aren't comparable (cosine vs BM25 vs no
score at all). Each item's fused score is

    rrf(d) = Σ_channel  1 / (k + rank_channel(d))

with ``k = 60`` as the standard smoothing constant. The function is
rank-only — it deliberately ignores per-channel scores so a channel
that returns calibrated similarity isn't penalised for "small" numbers
relative to a channel that returns idf-style weights.

We accept a list of ``(channel_id, ordered_ids)`` so the orchestrator
doesn't need to bundle channel objects into RRF — channels can be
arbitrary opaque labels (vector_entity, bm25, cypher, …).
"""

from __future__ import annotations

from typing import Dict, Hashable, Iterable, List, Sequence, Tuple


def reciprocal_rank_fusion(
    channel_rankings: Iterable[Tuple[Hashable, Sequence[str]]],
    *,
    k: int = 60,
    top_n: int | None = None,
) -> List[Tuple[str, float]]:
    """Fuse rankings from multiple channels via RRF.

    Parameters
    ----------
    channel_rankings:
        Iterable of ``(channel_id, ordered_ids)`` pairs. Each
        ``ordered_ids`` is the channel's hits, best-first. ``channel_id``
        is opaque — it's only used to key per-channel rank dicts.
    k:
        RRF smoothing constant (default 60 per Cormack et al.). Larger
        ``k`` weakens the bias toward top-ranked items.
    top_n:
        Optional cap. ``None`` returns the full fused list, ordered.

    Returns
    -------
    list[tuple[str, float]]
        ``(item_id, fused_score)`` pairs, sorted by fused score
        descending. Ties broken by first-seen order across channels —
        Python's sort is stable, so feeding the channels in a
        deterministic order gives reproducible output.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1 for RRF, got {k}")

    fused: Dict[str, float] = {}
    first_seen: Dict[str, int] = {}
    counter = 0

    for _channel_id, ordered in channel_rankings:
        for rank, item_id in enumerate(ordered, start=1):
            if not item_id:
                continue
            fused[item_id] = fused.get(item_id, 0.0) + 1.0 / (k + rank)
            if item_id not in first_seen:
                first_seen[item_id] = counter
                counter += 1

    # Sort by fused score desc, then by first-seen order for stability.
    items = sorted(
        fused.items(),
        key=lambda pair: (-pair[1], first_seen.get(pair[0], 0)),
    )
    if top_n is not None:
        items = items[:top_n]
    return items


__all__ = ["reciprocal_rank_fusion"]
