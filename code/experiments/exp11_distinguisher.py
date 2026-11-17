#!/usr/bin/env python3
"""
E11 -- Is the backdoored target stream distinguishable from an honest one?

E9 reports a single two-sample Kolmogorov--Smirnov test between one honest and
one backdoored stream of $20{,}000$ template targets, and that single test lands
close to the conventional threshold.  A single test cannot answer the question:
at this sample size the KS statistic is sensitive enough that two *honest*
streams also cross the threshold a few percent of the time, which is exactly
what a 5% test is supposed to do.  What matters to an auditor is whether the
test separates the two hypotheses better than it separates two draws of the same
hypothesis.

So we run the test against its own null.  For each of R independent replicates:

  * draw a fresh pairing root rho_A, a second fresh root rho_B, a fresh vendor
    key k_v and a fresh session id;
  * form three streams of S targets each: honest_A = Derive(rho_A, .),
    honest_B = Derive(rho_B, .), backdoor = PRF_{k_v}(sid, .);
  * compute KS(honest_A, honest_B)   -- the NULL: two honest streams;
  * compute KS(honest_A, backdoor)   -- the ALTERNATIVE.

If the backdoor is undetectable by this test, the two p-value distributions are
the same and the two rejection rates are both about alpha.  We report both
rejection rates with Wilson intervals, a two-proportion z-test between them, and
a KS test comparing the two p-value distributions themselves.  We do the same
for the chi-square goodness-of-fit statistic.

A rejection rate for the alternative that is materially above the null's would
falsify the paper's claim that the attack is invisible to distributional
inspection; a rate indistinguishable from the null supports it.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import Run, SEEDS
from pact.adversary import vendor_target
from pact.dpf import DPFParams, derive_template_from_slotkey
from pact.primitives import DeterministicRNG, slot_key
from pact.stats import chi_square_uniform, two_sample_ks, wilson_ci

R = 120          # independent replicates
S = 20000        # targets per stream
ALPHA = 0.05
PARAMS = DPFParams(n=16, m=32)
BINS = 128

SEEDS.setdefault("exp11_distinguisher", 20240011)


def honest_targets(root: bytes, n: int, params: DPFParams):
    return [derive_template_from_slotkey(slot_key(root, j), j, params).r
            for j in range(n)]


def main():
    name = "exp11_distinguisher"
    with Run(name, {"replicates": R, "stream_length": S, "alpha": ALPHA,
                    "bins": BINS,
                    "params": f"n={PARAMS.n},m={PARAMS.m}"}) as run:
        rng = DeterministicRNG(SEEDS[name])
        rows = []
        null_p, alt_p = [], []
        null_chi, alt_chi = [], []

        run.log(f"running {R} replicates of {S}-sample streams "
                f"(this measures the test against its own null)")
        for rep in range(R):
            rho_a, rho_b = rng.bytes(32), rng.bytes(32)
            kv, sid = rng.bytes(32), rng.bytes(16)
            ha = honest_targets(rho_a, S, PARAMS)
            hb = honest_targets(rho_b, S, PARAMS)
            bd = [vendor_target(kv, sid, j, PARAMS) for j in range(S)]

            _, p0 = two_sample_ks(ha, hb)
            _, p1 = two_sample_ks(ha, bd)
            null_p.append(p0)
            alt_p.append(p1)

            for tag, xs, acc in (("null", hb, null_chi), ("alt", bd, alt_chi)):
                counts = [0] * BINS
                for v in xs:
                    counts[v * BINS // PARAMS.domain] += 1
                acc.append(chi_square_uniform(counts)[2])

            if (rep + 1) % 50 == 0:
                run.log(f"  {rep+1}/{R} replicates done")

        n_null = sum(1 for p in null_p if p < ALPHA)
        n_alt = sum(1 for p in alt_p if p < ALPHA)
        lo0, hi0 = wilson_ci(n_null, R)
        lo1, hi1 = wilson_ci(n_alt, R)
        run.log(f"KS rejection rate at alpha={ALPHA}: "
                f"null (honest vs honest) {n_null}/{R} = {n_null/R:.3f} "
                f"[{lo0:.3f},{hi0:.3f}]; "
                f"alternative (honest vs backdoored) {n_alt}/{R} = {n_alt/R:.3f} "
                f"[{lo1:.3f},{hi1:.3f}]")

        # two-proportion z-test between the two rejection rates
        from math import sqrt
        pbar = (n_null + n_alt) / (2 * R)
        se = sqrt(max(1e-18, 2 * pbar * (1 - pbar) / R))
        z = (n_alt - n_null) / R / se if se > 0 else 0.0
        try:
            from scipy.stats import norm
            pz = 2 * (1 - norm.cdf(abs(z)))
        except Exception:
            pz = float("nan")
        run.log(f"two-proportion z-test on the rejection rates: z={z:.3f}, p={pz:.3f}")

        # do the two p-value distributions differ?
        d_pp, p_pp = two_sample_ks(null_p, alt_p)
        run.log(f"KS between the two p-value distributions: D={d_pp:.4f}, p={p_pp:.3f}")

        n_null_chi = sum(1 for p in null_chi if p < ALPHA)
        n_alt_chi = sum(1 for p in alt_chi if p < ALPHA)
        lo2, hi2 = wilson_ci(n_null_chi, R)
        lo3, hi3 = wilson_ci(n_alt_chi, R)
        run.log(f"chi-square rejection rate at alpha={ALPHA}: "
                f"null {n_null_chi}/{R} = {n_null_chi/R:.3f} [{lo2:.3f},{hi2:.3f}]; "
                f"alternative {n_alt_chi}/{R} = {n_alt_chi/R:.3f} "
                f"[{lo3:.3f},{hi3:.3f}]")

        rows.append({"test": "KS", "arm": "null (honest vs honest)",
                     "replicates": R, "rejections": n_null,
                     "rate": round(n_null / R, 5), "ci_lo": round(lo0, 5),
                     "ci_hi": round(hi0, 5),
                     "note": f"stream length {S}, alpha {ALPHA}"})
        rows.append({"test": "KS", "arm": "alternative (honest vs backdoored)",
                     "replicates": R, "rejections": n_alt,
                     "rate": round(n_alt / R, 5), "ci_lo": round(lo1, 5),
                     "ci_hi": round(hi1, 5),
                     "note": f"two-proportion z={z:.3f}, p={pz:.3f}"})
        rows.append({"test": "chi2", "arm": "null (honest)", "replicates": R,
                     "rejections": n_null_chi, "rate": round(n_null_chi / R, 5),
                     "ci_lo": round(lo2, 5), "ci_hi": round(hi2, 5),
                     "note": f"{BINS} bins"})
        rows.append({"test": "chi2", "arm": "alternative (backdoored)",
                     "replicates": R, "rejections": n_alt_chi,
                     "rate": round(n_alt_chi / R, 5), "ci_lo": round(lo3, 5),
                     "ci_hi": round(hi3, 5), "note": f"{BINS} bins"})
        rows.append({"test": "KS-of-p-values", "arm": "null vs alternative",
                     "replicates": R, "rejections": "", "rate": round(d_pp, 5),
                     "ci_lo": "", "ci_hi": "",
                     "note": f"D={d_pp:.4f}, p={p_pp:.3f}"})

        # keep the per-replicate p-values so the figure can show the whole
        # distribution rather than just the rejection counts
        from common import write_csv
        write_csv("exp11_pvalues",
                  [{"rep": i, "ks_null": round(null_p[i], 6),
                    "ks_alt": round(alt_p[i], 6),
                    "chi2_null": round(null_chi[i], 6),
                    "chi2_alt": round(alt_chi[i], 6)} for i in range(R)])
        run.log(f"wrote per-replicate p-values for {R} replicates")

        verdict = (abs(n_alt - n_null) / R < 0.05) and (p_pp > ALPHA)
        run.log(f"VERDICT: the backdoored stream is {'NOT ' if verdict else ''}"
                f"distinguishable from an honest stream by these tests")

        run.finish(rows, summary={
            "ks_reject_null": n_null / R, "ks_reject_alt": n_alt / R,
            "ks_two_proportion_z": round(z, 4), "ks_two_proportion_p": round(pz, 4),
            "pvalue_dist_ks_D": round(d_pp, 5), "pvalue_dist_ks_p": round(p_pp, 4),
            "chi2_reject_null": n_null_chi / R, "chi2_reject_alt": n_alt_chi / R,
            "indistinguishable": bool(verdict),
        })


if __name__ == "__main__":
    main()
