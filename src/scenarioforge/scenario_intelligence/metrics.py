"""Distribution metrics used by diversity, rarity, and curation evaluation.

All functions take mappings of category -> non-negative count (or weight) and are pure.

Definitions (p_i = count_i / sum(count)):
  * Shannon entropy      H = -sum_i p_i log2 p_i                  (bits; higher = more even)
  * Normalized entropy   H / log2(K), K = support size            (0..1; 1 = perfectly even)
  * Simpson diversity    1 - sum_i p_i^2                          (0..1; chance two draws differ)
  * Herfindahl index     sum_i p_i^2                              (higher = more concentrated)
  * Jensen-Shannon div.  0.5 KL(P||M) + 0.5 KL(Q||M), M=(P+Q)/2  (0..1 in base 2; 0 = identical)
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


def to_probabilities(counts: Mapping[Any, float]) -> dict[Any, float]:
    """Normalise counts to probabilities; an empty or all-zero mapping yields ``{}``."""
    if any(value < 0 for value in counts.values()):
        raise ValueError("Counts must be non-negative")
    total = sum(counts.values())
    if total <= 0:
        return {}
    return {key: value / total for key, value in counts.items() if value > 0}


def shannon_entropy(counts: Mapping[Any, float], base: float = 2.0) -> float:
    """Shannon entropy of a count distribution (0.0 for an empty distribution)."""
    probabilities = to_probabilities(counts)
    return -sum(p * math.log(p, base) for p in probabilities.values()) + 0.0


def normalized_entropy(counts: Mapping[Any, float], support_size: int | None = None) -> float:
    """Entropy divided by its maximum, log(K).

    ``support_size`` defaults to the number of observed categories. Pass the full taxonomy size
    to penalise categories that never occur. Distributions with K <= 1 return 0.0.
    """
    support = support_size if support_size is not None else len(to_probabilities(counts))
    if support <= 1:
        return 0.0
    return min(1.0, shannon_entropy(counts) / math.log2(support))


def simpson_diversity(counts: Mapping[Any, float]) -> float:
    """Gini-Simpson index 1 - sum p^2."""
    probabilities = to_probabilities(counts)
    return 1.0 - sum(p * p for p in probabilities.values()) if probabilities else 0.0


def herfindahl_index(counts: Mapping[Any, float]) -> float:
    """Concentration index sum p^2 (1.0 = a single category holds everything)."""
    return sum(p * p for p in to_probabilities(counts).values())


def top_k_share(counts: Mapping[Any, float], k: int = 3) -> float:
    """Fraction of the total held by the k most frequent categories."""
    probabilities = sorted(to_probabilities(counts).values(), reverse=True)
    return sum(probabilities[:k])


def imbalance_ratio(counts: Mapping[Any, float]) -> float | None:
    """Most frequent / least frequent non-zero count; ``None`` when undefined."""
    positive = [value for value in counts.values() if value > 0]
    if not positive:
        return None
    return max(positive) / min(positive)


def jensen_shannon_divergence(
    counts_p: Mapping[Any, float], counts_q: Mapping[Any, float]
) -> float:
    """Jensen-Shannon divergence in bits, bounded to [0, 1].

    Lower means the two distributions are more alike; 0 means identical. Either input being
    empty makes the divergence undefined, so ``ValueError`` is raised.
    """
    p, q = to_probabilities(counts_p), to_probabilities(counts_q)
    if not p or not q:
        raise ValueError("Jensen-Shannon divergence needs two non-empty distributions")
    divergence = 0.0
    for key in set(p) | set(q):
        p_i, q_i = p.get(key, 0.0), q.get(key, 0.0)
        mid = 0.5 * (p_i + q_i)
        if p_i > 0:
            divergence += 0.5 * p_i * math.log2(p_i / mid)
        if q_i > 0:
            divergence += 0.5 * q_i * math.log2(q_i / mid)
    return max(0.0, min(1.0, divergence))


def coverage(observed: Iterable[Any], reference: Iterable[Any]) -> float | None:
    """Fraction of reference categories that appear in ``observed`` (None if no reference)."""
    reference_set = set(reference)
    if not reference_set:
        return None
    return len(reference_set & set(observed)) / len(reference_set)


def inverse_frequency(frequency: float, epsilon: float = 1.0) -> float:
    """Rarity score 1 / (frequency + epsilon); epsilon must be positive."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if frequency < 0:
        raise ValueError("frequency must be non-negative")
    return 1.0 / (frequency + epsilon)


def min_max_normalize(values: Mapping[str, float | None]) -> dict[str, float | None]:
    """Scale values to [0, 1]; ``None`` is preserved and a constant series maps to 0.0."""
    present = [v for v in values.values() if v is not None]
    if not present:
        return {key: None for key in values}
    low, high = min(present), max(present)
    span = high - low
    return {
        key: None if value is None else ((value - low) / span if span > 0 else 0.0)
        for key, value in values.items()
    }


def distribution_summary(
    counts: Mapping[Any, float], *, support_size: int | None = None, top_k: int = 3
) -> dict[str, Any]:
    """Standard diversity/concentration summary of one distribution."""
    positive = {key: value for key, value in counts.items() if value > 0}
    return {
        "total": sum(positive.values()),
        "observed_categories": len(positive),
        "shannon_entropy_bits": shannon_entropy(positive),
        "normalized_entropy": normalized_entropy(positive, support_size),
        "simpson_diversity": simpson_diversity(positive),
        "herfindahl_index": herfindahl_index(positive),
        f"top_{top_k}_share": top_k_share(positive, top_k),
        "imbalance_ratio": imbalance_ratio(positive),
    }


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity; two zero vectors are treated as identical (1.0), one zero as 0.0."""
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 and norm_b == 0:
        return 1.0
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (norm_a * norm_b)


def jaccard_similarity(a: Iterable[Any], b: Iterable[Any]) -> float:
    """Jaccard similarity of two sets; two empty sets are identical (1.0)."""
    set_a, set_b = set(a), set(b)
    union = set_a | set_b
    return len(set_a & set_b) / len(union) if union else 1.0
