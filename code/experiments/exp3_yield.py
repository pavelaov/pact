#!/usr/bin/env python3
"""
E3 -- What the deterrent costs: the detection/yield frontier.

An audited slot is *sacrificed*, so the auditor buys detection with key yield.
This experiment measures, over real batches:
  * the realised sacrifice fraction against the nominal audit rate p
    (the coin must not be biased, or the auditor's dial would not mean what
    it says);
  * the achievable (yield, detection) frontier for several leak sizes c;
  * the smallest p that reaches a target detection level for a given c
    (the auditor's provisioning rule).
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import Run, SEEDS
from pact.audit import challenge_set, coin_toss
from pact.primitives import DeterministicRNG, h
from pact.stats import mean_ci, wilson_ci

N_GRID = (64, 256, 1024)
P_GRID = (0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50)
TRIALS = 5000
C_FRONTIER = (1, 4, 8, 16, 32, 64)
TARGETS = (0.5, 0.9, 0.99, 0.999)


def main():
    name = "exp3_yield"
    with Run(name, {"N_grid": list(N_GRID), "p_grid": list(P_GRID),
                    "trials": TRIALS, "c_frontier": list(C_FRONTIER),
                    "targets": list(TARGETS)}) as run:
        rows = []

        # (a) is the realised sacrifice fraction the nominal p?
        for N in N_GRID:
            counters = list(range(N))
            for p in P_GRID:
                rng = DeterministicRNG(f"{SEEDS[name]}|{N}|{p}")
                fracs = []
                for _ in range(TRIALS):
                    coin = coin_toss(h(b"br", rng.bytes(16)),
                                     DeterministicRNG(rng.bytes(16)),
                                     DeterministicRNG(rng.bytes(16)))
                    fracs.append(len(challenge_set(coin.chi, counters, p)) / N)
                mu, lo, hi = mean_ci(fracs)
                rel = abs(mu - p) / p
                rows.append({"part": "sacrifice", "N": N, "p": p, "c": "",
                             "trials": TRIALS, "sacrifice_mean": round(mu, 6),
                             "sacrifice_ci_lo": round(lo, 6),
                             "sacrifice_ci_hi": round(hi, 6),
                             "yield_mean": round(1 - mu, 6),
                             "rel_err_vs_p": round(rel, 6),
                             "detect": "", "analytic_detect": ""})
                run.log(f"N={N:<5} p={p:<5} sacrificed={mu:.5f} "
                        f"[{lo:.5f},{hi:.5f}] yield={1-mu:.5f} rel.err={rel:.2%}")

        worst = max(r["rel_err_vs_p"] for r in rows if r["part"] == "sacrifice")
        run.log(f"worst relative deviation of realised sacrifice from nominal p: {worst:.3%}")

        # (b) the frontier itself
        for c in C_FRONTIER:
            for p in P_GRID:
                det = 1 - (1 - p) ** c
                rows.append({"part": "frontier", "N": "", "p": p, "c": c,
                             "trials": "", "sacrifice_mean": "",
                             "sacrifice_ci_lo": "", "sacrifice_ci_hi": "",
                             "yield_mean": round(1 - p, 6), "rel_err_vs_p": "",
                             "detect": round(det, 6), "analytic_detect": round(det, 6)})

        # (c) the auditor's provisioning rule: smallest p for a target detection
        run.log("--- provisioning: smallest p reaching a target detection level")
        for c in C_FRONTIER:
            for tgt in TARGETS:
                p_star = 1 - (1 - tgt) ** (1.0 / c)
                rows.append({"part": "provision", "N": "", "p": round(p_star, 6),
                             "c": c, "trials": "", "sacrifice_mean": "",
                             "sacrifice_ci_lo": "", "sacrifice_ci_hi": "",
                             "yield_mean": round(1 - p_star, 6), "rel_err_vs_p": "",
                             "detect": tgt, "analytic_detect": tgt})
                run.log(f"c={c:<3} target={tgt:<6} -> p*={p_star:.4f} "
                        f"(yield {1-p_star:.4f})")

        run.finish(rows, summary={"worst_rel_sacrifice_err": round(worst, 6)})


if __name__ == "__main__":
    main()
