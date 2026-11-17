#!/usr/bin/env python3
"""
E5 -- Defamation freeness under an actively malicious host.

The audit is run by the parties, who are untrusted.  A host that wants to
destroy an honest chip's reputation controls the whole transcript on the way
to a third party.  This experiment enumerates nine concrete forgery strategies
and, for each, runs many independent attempts against an honest chip, checking
that the third-party ``judgement`` procedure convicts in exactly zero of them.

It also runs the positive control: a genuinely cheating chip, whose
certificate ``judgement`` must accept.  Without the control, a judge that
always says "not guilty" would pass this test.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import Run, SEEDS
from pact.adversary import blind_taint_choice
from pact.audit import (FORGERY_STRATEGIES, audit_batch, forge_certificate,
                        judgement)
from pact.dpf import DPFParams
from pact.primitives import DeterministicRNG
from pact.protocol import CheatProfile, Chip, pair_chips
from pact.stats import rule_of_three, wilson_ci

N_BATCH = 128
TRIALS = 1000            # per strategy
CONTROL_TRIALS = 1000
P = 0.10
PARAMS = DPFParams(n=16, m=32)


def main():
    name = "exp5_defamation"
    with Run(name, {"N": N_BATCH, "trials_per_strategy": TRIALS,
                    "control_trials": CONTROL_TRIALS, "p": P,
                    "strategies": list(FORGERY_STRATEGIES)}) as run:
        counters = list(range(N_BATCH))
        rows = []
        total_convictions = 0

        for strat in FORGERY_STRATEGIES:
            rng = DeterministicRNG(f"{SEEDS[name]}|{strat}")
            ctx, keys = pair_chips(rng, PARAMS)
            convicted = 0
            rejected_because = {}
            for t in range(TRIALS):
                chip = Chip(0, ctx, keys[0])     # HONEST chip
                chip.gen_batch(counters)
                cert = forge_certificate(chip, counters, P,
                                         DeterministicRNG(rng.bytes(16)),
                                         DeterministicRNG(rng.bytes(16)), strat)
                guilty, why = judgement(cert, ctx.attest_pub[0], PARAMS)
                convicted += guilty
                if not guilty:
                    rejected_because[why] = rejected_because.get(why, 0) + 1
            total_convictions += convicted
            lo, hi = wilson_ci(convicted, TRIALS)
            top = max(rejected_because.items(), key=lambda kv: kv[1])[0] if rejected_because else ""
            rows.append({"part": "forgery", "strategy": strat, "trials": TRIALS,
                         "convictions": convicted,
                         "conviction_rate": round(convicted / TRIALS, 6),
                         "ci_lo": round(lo, 6), "ci_hi": round(hi, 6),
                         "dominant_rejection_reason": top})
            run.log(f"{strat:<26} convictions {convicted}/{TRIALS}  "
                    f"[{lo:.4f},{hi:.4f}]  <- {top}")

        n_attempts = TRIALS * len(FORGERY_STRATEGIES)
        run.log(f"TOTAL forged accusations upheld: {total_convictions}/{n_attempts} "
                f"(95% upper bound {rule_of_three(n_attempts):.2e})")

        # positive control -- a really cheating chip must be convicted
        rng = DeterministicRNG(f"{SEEDS[name]}|control")
        ctx, keys = pair_chips(rng, PARAMS)
        upheld = caught = 0
        for t in range(CONTROL_TRIALS):
            plan = blind_taint_choice(counters, 32, rng)
            chip = Chip(0, ctx, keys[0],
                        cheat=CheatProfile(frozenset(plan.taints), "seed"),
                        rng=DeterministicRNG(rng.bytes(16)))
            chip.gen_batch(counters)
            rep = audit_batch(chip, counters, 0.5, DeterministicRNG(rng.bytes(16)),
                              DeterministicRNG(rng.bytes(16)))
            if rep.caught:
                caught += 1
                upheld += judgement(rep.certificate, ctx.attest_pub[0], PARAMS)[0]
        lo, hi = wilson_ci(upheld, caught) if caught else (0, 1)
        rows.append({"part": "control", "strategy": "genuine-cheat", "trials": caught,
                     "convictions": upheld,
                     "conviction_rate": round(upheld / caught, 6) if caught else 0,
                     "ci_lo": round(lo, 6), "ci_hi": round(hi, 6),
                     "dominant_rejection_reason": ""})
        run.log(f"positive control: {caught}/{CONTROL_TRIALS} audits caught a real "
                f"cheat; {upheld}/{caught} certificates upheld by an independent judge")

        run.finish(rows, summary={
            "forged_attempts": n_attempts, "forged_upheld": total_convictions,
            "false_conviction_upper_95": rule_of_three(n_attempts),
            "control_caught": caught, "control_upheld": upheld})


if __name__ == "__main__":
    main()
