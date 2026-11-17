#!/usr/bin/env python3
"""
E6 -- Input consistency against a malicious host.

The audit constrains the chip.  A symmetric commitment constrains the host:
at submit time the host commits to (alpha_b, beta_b); at bind time the chip
recomputes the commitment from the share it is actually given and aborts,
with attribution, on any mismatch.

Arms:
  honest            baseline; must never abort
  shift-alpha       host binds alpha_b + 1
  shift-beta        host binds a different payload share
  swap-opening      host binds the share it committed for a *different* slot
  replay-old-slot   host reuses last slot's opening on this slot
  no-commitment     ablation: the mechanism is switched off, so the same
                    attacks succeed silently -- this is the number that shows
                    the commitment is load-bearing rather than decorative

For every arm we also record whether the *key that would have been produced*
differs from the honest one, i.e. whether the attack would have had an effect
had it not been caught.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import Run, SEEDS
from pact.dpf import DPFParams, dpf_eval_full
from pact.primitives import DeterministicRNG
from pact.protocol import (Chip, Host, InputConsistencyAbort, eval_key,
                           ic_commit, pair_chips)
from pact.stats import clopper_pearson, rule_of_three

TRIALS = 3000
PARAMS = DPFParams(n=12, m=32)
ARMS = ("honest", "shift-alpha", "shift-beta", "swap-opening",
        "replay-old-slot", "no-commitment")


def one_trial(ctx, keys, rng, cnt, arm, enforce):
    p = ctx.params
    c0, c1 = Chip(0, ctx, keys[0]), Chip(1, ctx, keys[1])
    c0.gen(cnt); c1.gen(cnt)
    h0 = Host(0, ctx, DeterministicRNG(rng.bytes(16)))
    h1 = Host(1, ctx, DeterministicRNG(rng.bytes(16)))

    alpha, beta = rng.randint(p.domain), rng.randint(p.modulus)
    a0 = rng.randint(p.domain); a1 = (alpha - a0) % p.domain
    b0 = rng.randint(p.modulus); b1 = (beta - b0) % p.modulus

    cm0, op0 = h0.submit(cnt, a0, b0)
    cm1, op1 = h1.submit(cnt, a1, b1)

    ia0, ib0, nu0 = op0
    if arm == "shift-alpha":
        ia0 = (ia0 + 1) % p.domain
    elif arm == "shift-beta":
        ib0 = (ib0 + 7) % p.modulus
    elif arm == "swap-opening":
        cm_other, op_other = h0.submit(cnt + 1, (a0 + 3) % p.domain, b0)
        ia0, ib0, nu0 = op_other
    elif arm == "replay-old-slot":
        cm_old, op_old = h0.submit(cnt - 1 if cnt else cnt + 1, a0, b0)
        nu0 = op_old[2]
    elif arm == "no-commitment":
        ia0 = (ia0 + 1) % p.domain

    m0 = c0.bind_message(cnt, ia0, ib0, cm0)
    m1 = c1.bind_message(cnt, op1[0], op1[1], cm1)

    kw0 = dict(my_ic_opening=(ia0, ib0, nu0), my_ic_commit=cm0) if enforce else {}
    kw1 = dict(my_ic_opening=op1, my_ic_commit=cm1) if enforce else {}
    try:
        d0, k0 = c0.bind_finish(cnt, ia0, ib0, m1, **kw0)
        d1, k1 = c1.bind_finish(cnt, op1[0], op1[1], m0, **kw1)
    except InputConsistencyAbort:
        return True, None
    # not caught: did the deviation actually change the key?
    if d0 != d1:
        return False, "shift-mismatch"
    f0, f1 = dpf_eval_full(k0), dpf_eval_full(k1)
    D = p.domain
    ok = all(((f0[(x + d0) % D] + f1[(x + d0) % D]) % p.modulus)
             == (beta if x == alpha else 0) for x in range(D))
    return False, ("key-unchanged" if ok else "key-corrupted")


def main():
    name = "exp6_consistency"
    with Run(name, {"trials": TRIALS, "arms": list(ARMS),
                    "params": f"n={PARAMS.n},m={PARAMS.m}"}) as run:
        rows = []
        for arm in ARMS:
            enforce = (arm != "no-commitment")
            rng = DeterministicRNG(f"{SEEDS[name]}|{arm}")
            ctx, keys = pair_chips(rng, PARAMS)
            caught = 0
            outcomes = {}
            for t in range(TRIALS):
                c, outcome = one_trial(ctx, keys, rng, t + 1, arm, enforce)
                caught += c
                if not c:
                    outcomes[outcome] = outcomes.get(outcome, 0) + 1
            lo, hi = clopper_pearson(caught, TRIALS)
            rows.append({"arm": arm, "enforced": int(enforce), "trials": TRIALS,
                         "aborted_with_attribution": caught,
                         "detect_rate": round(caught / TRIALS, 6),
                         "ci_lo": round(lo, 6), "ci_hi": round(hi, 6),
                         "undetected_outcomes": ";".join(
                             f"{k}={v}" for k, v in sorted(outcomes.items(), key=str))})
            run.log(f"{arm:<18} enforce={int(enforce)} caught {caught}/{TRIALS} "
                    f"[{lo:.4f},{hi:.4f}] undetected={outcomes}")

        run.finish(rows, summary={
            "miss_upper_95_per_arm": rule_of_three(TRIALS),
            "note": "the no-commitment arm is the ablation: with the mechanism "
                    "removed the identical deviation is never detected"})


if __name__ == "__main__":
    main()
