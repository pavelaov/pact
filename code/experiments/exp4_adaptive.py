#!/usr/bin/env python3
"""
E4 -- Does the commit-then-challenge ordering actually buy anything?

Three arms, identical in every other respect:

  BLIND   the real protocol.  The chip commits its whole batch and signs the
          batch root; only then do the hosts run the coin.  The chip's choice
          of which slots to taint is independent of the challenge.

  ORACLE  a counterfactual chip that learns the challenge *before* it commits
          and places every taint on a slot it knows will not be opened.  This
          is what an adversary gets if the coin is tossed first, i.e. if the
          ordering is reversed.

  BIASED  a weaker but more realistic ablation: one *malicious host* tries to
          steer the coin by choosing its opening after seeing the peer's.
          With commitments this is impossible; without them the host can
          grind its contribution.  We run the no-commitment variant to
          measure how much grinding buys.

The gap between BLIND and ORACLE is the value of the ordering.  The gap
between BLIND and BIASED is the value of the *commitment* inside the coin.
"""

from __future__ import annotations

import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import Run, SEEDS
from pact.adversary import blind_taint_choice, oracle_taint_choice
from pact.audit import (audit_batch, challenge_set, coin_toss, judgement,
                        slot_is_marked)
from pact.dpf import DPFParams
from pact.primitives import DeterministicRNG, h
from pact.protocol import CheatProfile, Chip, pair_chips
from pact.stats import wilson_ci

N_BATCH = 256
TRIALS = 20000
PIPE_TRIALS = 300
P_GRID = (0.05, 0.10, 0.20)
C_GRID = (1, 4, 8, 16, 32)
GRIND_TRIES = 64          # how many openings a coin-grinding host may try
PARAMS = DPFParams(n=16, m=32)


def main():
    name = "exp4_adaptive"
    with Run(name, {"N": N_BATCH, "trials": TRIALS, "pipeline_trials": PIPE_TRIALS,
                    "p_grid": list(P_GRID), "c_grid": list(C_GRID),
                    "grind_tries": GRIND_TRIES}) as run:
        counters = list(range(N_BATCH))
        idx = {c: i for i, c in enumerate(counters)}
        rows = []

        for p in P_GRID:
            for c in C_GRID:
                rng = DeterministicRNG(f"{SEEDS[name]}|{p}|{c}")
                blind_hits = oracle_hits = grind_hits = 0
                for _ in range(TRIALS):
                    root = h(b"br", rng.bytes(16))
                    r0, r1 = DeterministicRNG(rng.bytes(16)), DeterministicRNG(rng.bytes(16))
                    coin = coin_toss(root, r0, r1)

                    # BLIND: taints chosen without knowledge of chi
                    plan = blind_taint_choice(counters, c, rng)
                    if any(slot_is_marked(coin.chi, idx[x], x, p) for x in plan.taints):
                        blind_hits += 1

                    # ORACLE: taints chosen with full knowledge of chi
                    marked = challenge_set(coin.chi, counters, p)
                    oplan = oracle_taint_choice(counters, marked, c)
                    if any(slot_is_marked(coin.chi, idx[x], x, p) for x in oplan.taints):
                        oracle_hits += 1

                    # BIASED: no commitment in the coin -> host 1 grinds its
                    # opening after seeing host 0's, minimising the audit set
                    best_chi, best_marked = None, None
                    for g in range(GRIND_TRIES):
                        cand = h(b"PACT|chi", root, coin.open0,
                                 rng.bytes(16))
                        m = challenge_set(cand, counters, p)
                        if best_marked is None or len(m) < len(best_marked):
                            best_chi, best_marked = cand, m
                    if any(slot_is_marked(best_chi, idx[x], x, p) for x in plan.taints):
                        grind_hits += 1

                ana = 1 - (1 - p) ** c
                for arm, hits in (("blind", blind_hits), ("oracle", oracle_hits),
                                  ("grind-no-commit", grind_hits)):
                    lo, hi = wilson_ci(hits, TRIALS)
                    rows.append({"arm": arm, "estimator": "monte-carlo", "p": p,
                                 "tainted": c, "trials": TRIALS, "detected": hits,
                                 "detect": round(hits / TRIALS, 6),
                                 "ci_lo": round(lo, 6), "ci_hi": round(hi, 6),
                                 "analytic_blind": round(ana, 6)})
                run.log(f"p={p:<5} c={c:<3} blind={blind_hits/TRIALS:.4f} "
                        f"oracle={oracle_hits/TRIALS:.4f} "
                        f"grind={grind_hits/TRIALS:.4f} analytic={ana:.4f}")

        # full-pipeline confirmation of the blind arm with real chips
        run.log("--- full-pipeline confirmation (real chips + third-party judgement)")
        for p, c in ((0.10, 16),):
            rng = DeterministicRNG(f"{SEEDS[name]}|pipe")
            ctx, keys = pair_chips(rng, PARAMS)
            hits = judged = 0
            for _ in range(PIPE_TRIALS):
                plan = blind_taint_choice(counters, c, rng)
                chip = Chip(0, ctx, keys[0],
                            cheat=CheatProfile(frozenset(plan.taints), "target"),
                            rng=DeterministicRNG(rng.bytes(16)))
                chip.gen_batch(counters)
                rep = audit_batch(chip, counters, p, DeterministicRNG(rng.bytes(16)),
                                  DeterministicRNG(rng.bytes(16)))
                if rep.caught:
                    hits += 1
                    judged += judgement(rep.certificate, ctx.attest_pub[0], PARAMS)[0]
            lo, hi = wilson_ci(hits, PIPE_TRIALS)
            ana = 1 - (1 - p) ** c
            rows.append({"arm": "blind", "estimator": "full-pipeline", "p": p,
                         "tainted": c, "trials": PIPE_TRIALS, "detected": hits,
                         "detect": round(hits / PIPE_TRIALS, 6),
                         "ci_lo": round(lo, 6), "ci_hi": round(hi, 6),
                         "analytic_blind": round(ana, 6)})
            run.log(f"pipeline p={p} c={c}: detect={hits/PIPE_TRIALS:.4f} "
                    f"[{lo:.4f},{hi:.4f}] analytic={ana:.4f}; "
                    f"certificates upheld {judged}/{hits}")
            assert judged == hits

        run.finish(rows, summary={"note": "oracle arm is counterfactual: it "
                                          "requires the challenge before the commitment"})


if __name__ == "__main__":
    main()
