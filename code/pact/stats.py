"""Statistical helpers: everything the paper reports comes with dispersion."""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion (95% by default).

    Preferred over the normal approximation because several of the paper's
    cells are at or extremely near 0 or 1, where the Wald interval is wrong.
    """
    if n == 0:
        return (0.0, 1.0)
    phat = k / n
    denom = 1.0 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Exact (conservative) binomial interval, used for the all-or-nothing cells."""
    from scipy.stats import beta  # type: ignore
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return (float(lo), float(hi))


def rule_of_three(n: int) -> float:
    """Upper 95% bound on a probability after n trials with zero events."""
    return 3.0 / n if n > 0 else 1.0


def mean_std(xs: Sequence[float]) -> Tuple[float, float]:
    n = len(xs)
    if n == 0:
        return (0.0, 0.0)
    mu = sum(xs) / n
    if n == 1:
        return (mu, 0.0)
    var = sum((x - mu) ** 2 for x in xs) / (n - 1)
    return (mu, math.sqrt(var))


def mean_ci(xs: Sequence[float], z: float = 1.959963984540054) -> Tuple[float, float, float]:
    mu, sd = mean_std(xs)
    half = z * sd / math.sqrt(len(xs)) if xs else 0.0
    return (mu, mu - half, mu + half)


def chi_square_uniform(counts: Sequence[int]) -> Tuple[float, int, float]:
    """Chi-square goodness of fit against a uniform distribution."""
    from scipy.stats import chi2  # type: ignore
    n = sum(counts)
    k = len(counts)
    exp = n / k
    stat = sum((c - exp) ** 2 / exp for c in counts)
    dof = k - 1
    return (stat, dof, float(chi2.sf(stat, dof)))


def two_sample_ks(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float]:
    from scipy.stats import ks_2samp  # type: ignore
    r = ks_2samp(list(a), list(b))
    return (float(r.statistic), float(r.pvalue))


def entropy_bits_uniform_over(k: int) -> float:
    return math.log2(k) if k > 0 else 0.0


def geometric_stats(p: float) -> Tuple[float, float]:
    """Mean and variance of the number of successful cheats before first catch."""
    if p <= 0:
        return (float("inf"), float("inf"))
    return ((1 - p) / p, (1 - p) / (p * p))
