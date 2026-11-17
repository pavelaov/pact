#!/usr/bin/env python3
"""
E2 -- The deterrent factor:  P_detect(c,p) = 1 - (1-p)^c.

Two independent estimators of the same quantity, so the closed form in the
paper is validated rather than assumed:

  * MONTE-CARLO over the *real* mechanism.  Each trial builds a fresh
    commit-then-challenge coin from fresh host randomness, expands it with the
    real ``slot_is_marked`` PRF, and asks whether any of ``c`` blindly chosen
    tainted slots was marked.  20,000 trials per cell.

  * FULL-PIPELINE CROSS-CHECK.  A smaller number of trials that build real
    Chip objects, run GEN over a whole batch with a real cheat profile, run
    the real ``audit_batch``, and additionally verify every emitted
    certificate with the third-party ``judgement``.  This confirms that the
    cheap estimator above is faithful to the complete protocol.
"""

from __future__ import annotations

import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import Run, SEEDS
from pact.adversary import blind_taint_choice
from pact.audit import audit_batch, coin_toss, judgement, slot_is_marked
from pact.dpf import DPFParams
from pact.primitives import DeterministicRNG, h
from pact.protocol import CheatProfile, Chip, pair_chips
from pact.stats import wilson_ci

N_BATCH = 256
TRIALS = 20000
XCHECK_TRIALS = 400
P_GRID = (0.02, 0.05, 0.10, 0.20, 0.50)
C_GRID = (1, 2, 4, 8, 16, 32, 64)
PARAMS = DPFParams(n=16, m=32)


def analytic(c: int, p: float) -> float:
    return 1.0 - (1.0 - p) ** c


def main():
    name = "exp2_detection"
    with Run(name, {"N": N_BATCH, "trials_per_cell": TRIALS,
                    "xcheck_trials": XCHECK_TRIALS,
                    "p_grid": list(P_GRID), "c_grid": list(C_GRID)}) as run:
        counters = list(range(N_BATCH))
        rows = []
        max_err = 0.0

        for p in P_GRID:
            for c in C_GRID:
                rng = DeterministicRNG(f"{SEEDS[name]}|{p}|{c}")
                hits = 0
                for t in range(TRIALS):
                    # a fresh batch root and a fresh, real commit-then-challenge coin
                    batch_root = h(b"batchroot", rng.bytes(16))
                    coin = coin_toss(batch_root,
                                     DeterministicRNG(rng.bytes(16)),
                                     DeterministicRNG(rng.bytes(16)))
                    # the chip commits its taints BEFORE seeing chi
                    plan = blind_taint_choice(counters, c, rng)
                    idx = {cnt: i for i, cnt in enumerate(counters)}
                    if any(slot_is_marked(coin.chi, idx[cnt], cnt, p) for cnt in plan.taints):
                        hits += 1
                emp = hits / TRIALS
                ana = analytic(c, p)
                lo, hi = wilson_ci(hits, TRIALS)
                max_err = max(max_err, abs(emp - ana))
                rows.append({"estimator": "monte-carlo", "p": p, "tainted": c,
                             "trials": TRIALS, "detected": hits,
                             "emp_detect": round(emp, 6),
                             "ci_lo": round(lo, 6), "ci_hi": round(hi, 6),
                             "analytic_detect": round(ana, 6),
                             "abs_err": round(abs(emp - ana), 6),
                             "within_ci": int(lo <= ana <= hi),
                             "yield_kept": round(1 - p, 4)})
                run.log(f"p={p:<5} c={c:<3} emp={emp:.4f} [{lo:.4f},{hi:.4f}] "
                        f"ana={ana:.4f} err={abs(emp-ana):.4f} "
                        f"{'OK' if lo <= ana <= hi else 'OUT'}")

        run.log(f"max |empirical - analytic| over all cells = {max_err:.4f}")
        n_out = sum(1 for r in rows if not r["within_ci"])
        run.log(f"cells whose 95% CI excludes the closed form: {n_out}/{len(rows)}")

        # ------------------------------------------------------------------
        # full-pipeline cross-check
        # ------------------------------------------------------------------
        run.log("--- full-pipeline cross-check (real chips, real audit, real judgement)")
        for p, c in ((0.10, 8), (0.20, 4), (0.05, 16)):
            rng = DeterministicRNG(f"{SEEDS[name]}|xc|{p}|{c}")
            ctx, keys = pair_chips(rng, PARAMS)
            hits = 0
            judged = 0
            for t in range(XCHECK_TRIALS):
                plan = blind_taint_choice(counters, c, rng)
                chip = Chip(0, ctx, keys[0],
                            cheat=CheatProfile(frozenset(plan.taints), "seed"),
                            rng=DeterministicRNG(rng.bytes(16)))
                chip.gen_batch(counters)
                rep = audit_batch(chip, counters, p,
                                  DeterministicRNG(rng.bytes(16)),
                                  DeterministicRNG(rng.bytes(16)))
                if rep.caught:
                    hits += 1
                    guilty, _ = judgement(rep.certificate, ctx.attest_pub[0], PARAMS)
                    judged += guilty
            emp = hits / XCHECK_TRIALS
            ana = analytic(c, p)
            lo, hi = wilson_ci(hits, XCHECK_TRIALS)
            rows.append({"estimator": "full-pipeline", "p": p, "tainted": c,
                         "trials": XCHECK_TRIALS, "detected": hits,
                         "emp_detect": round(emp, 6), "ci_lo": round(lo, 6),
                         "ci_hi": round(hi, 6), "analytic_detect": round(ana, 6),
                         "abs_err": round(abs(emp - ana), 6),
                         "within_ci": int(lo <= ana <= hi),
                         "yield_kept": round(1 - p, 4)})
            run.log(f"p={p} c={c}: pipeline detect {emp:.4f} [{lo:.4f},{hi:.4f}] "
                    f"vs analytic {ana:.4f}; every catch verified by an independent "
                    f"third party: {judged}/{hits}")
            assert judged == hits, "a certificate failed public verification"

        run.finish(rows, summary={"max_abs_err": round(max_err, 6),
                                  "cells": len(P_GRID) * len(C_GRID),
                                  "cells_ci_excluding_analytic": n_out})


if __name__ == "__main__":
    main()
