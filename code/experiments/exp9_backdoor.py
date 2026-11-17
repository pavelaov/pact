#!/usr/bin/env python3
"""
E9 (NEW) -- An implemented backdoor that only an audit can see.

Attestation certifies which firmware a chip runs.  Correctness testing
certifies that its keys work.  Neither certifies that the keys were *derived*
honestly.  To make that gap concrete rather than rhetorical we implement the
strongest covert generator we know how to build against a template-based FSS
key generator and measure what it buys and what PACT takes away.

The attack.  For a chosen subset of slots the chip derives the template target
from a vendor key instead of from the pairing root:

    r_cnt = PRF_{k_v}(sid, cnt)                    instead of  Derive(rho_cnt)

Everything else is untouched.  The template is a valid payload-0 DPF, the
bound key is a correct DPF key, and every value the hosts see is
pseudorandom.  But the bind step publishes the shift Delta = r - alpha, so
anyone holding k_v recovers the secret index of the query:

    alpha = PRF_{k_v}(sid, cnt) - Delta.

Questions:
  Q1  Does the attack work?  (exact-recovery rate of alpha from public data)
  Q2  Is it invisible to everything except a derivation recompute?
      - do the keys still compute the right function? (full-domain check)
      - do backdoored targets look non-uniform? (chi-square)
      - are backdoored and honest target streams distinguishable? (two-sample
        Kolmogorov-Smirnov)
      - does the well-formedness half of the audit alone catch it?
  Q3  Does PACT catch it, and at what rate?
  Q4  How many queries does a vendor deanonymise before the first catch?
      (the operational reading of the deterrent: a leakage budget)
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import Run, SEEDS
from pact.adversary import (VendorBackdoorChip, blind_taint_choice,
                            vendor_recover_alpha, vendor_target)
from pact.audit import audit_batch, check_opened_slot, judgement
from pact.dpf import (DPFParams, derive_template_from_slotkey, dpf_eval_full,
                      template_is_wellformed)
from pact.primitives import DeterministicRNG, slot_key
from pact.protocol import Chip, Host, eval_key, pair_chips, run_keygen
from pact.stats import (chi_square_uniform, clopper_pearson, geometric_stats,
                        mean_ci, rule_of_three, two_sample_ks, wilson_ci)

N_BATCH = 256
Q1_TRIALS = 2000          # binds against a fully backdoored chip
Q2_KEYS = 300             # full-domain correctness of backdoored keys
Q2_SAMPLES = 20000        # targets for the distribution tests
Q3_TRIALS = 4000          # audits per (p, c) cell
Q4_TRIALS = 5000          # leakage-budget runs
P_GRID = (0.02, 0.05, 0.10, 0.20)
C_GRID = (1, 4, 16, 64)
PARAMS = DPFParams(n=16, m=32)
PARAMS_SMALL = DPFParams(n=12, m=32)
VENDOR_KEY = bytes.fromhex("5645" * 16)


def main():
    name = "exp9_backdoor"
    with Run(name, {"N": N_BATCH, "q1_trials": Q1_TRIALS, "q2_keys": Q2_KEYS,
                    "q2_samples": Q2_SAMPLES, "q3_trials": Q3_TRIALS,
                    "q4_trials": Q4_TRIALS, "p_grid": list(P_GRID),
                    "c_grid": list(C_GRID),
                    "params": f"n={PARAMS.n},m={PARAMS.m}"}) as run:
        rows = []
        counters = list(range(N_BATCH))

        # ================================================================ Q1
        run.log("Q1: does the backdoor recover the secret index from public data?")
        rng = DeterministicRNG(f"{SEEDS[name]}|q1")
        ctx, keys = pair_chips(rng, PARAMS)
        recovered = 0
        for t in range(Q1_TRIALS):
            cnt = t
            c0 = VendorBackdoorChip(0, ctx, keys[0], VENDOR_KEY, [cnt],
                                    DeterministicRNG(rng.bytes(16)))
            c1 = VendorBackdoorChip(1, ctx, keys[1], VENDOR_KEY, [cnt],
                                    DeterministicRNG(rng.bytes(16)))
            c0.gen(cnt); c1.gen(cnt)
            h0 = Host(0, ctx, DeterministicRNG(rng.bytes(16)))
            h1 = Host(1, ctx, DeterministicRNG(rng.bytes(16)))
            alpha = rng.randint(PARAMS.domain)
            beta = rng.randint(PARAMS.modulus)
            delta, k0, k1 = run_keygen(c0, c1, h0, h1, cnt, alpha, beta, rng)
            guess = vendor_recover_alpha(VENDOR_KEY, ctx.sid, cnt, delta, PARAMS)
            recovered += (guess == alpha)
        lo, hi = clopper_pearson(recovered, Q1_TRIALS)
        run.log(f"  exact recovery of alpha: {recovered}/{Q1_TRIALS} "
                f"= {recovered/Q1_TRIALS:.4f} [{lo:.4f},{hi:.4f}]; "
                f"chance floor 2^-{PARAMS.n} = {2.0**-PARAMS.n:.2e}")
        rows.append({"part": "Q1_recovery", "cell": "backdoored", "p": "", "c": "",
                     "trials": Q1_TRIALS, "value": recovered / Q1_TRIALS,
                     "lo": lo, "hi": hi,
                     "note": f"chance floor 2^-{PARAMS.n}={2.0**-PARAMS.n:.3e}"})

        # honest control: the same adversary against an honest chip
        rng = DeterministicRNG(f"{SEEDS[name]}|q1c")
        ctxh, keysh = pair_chips(rng, PARAMS)
        rec_h = 0
        for t in range(Q1_TRIALS):
            cnt = t
            c0, c1 = Chip(0, ctxh, keysh[0]), Chip(1, ctxh, keysh[1])
            c0.gen(cnt); c1.gen(cnt)
            h0 = Host(0, ctxh, DeterministicRNG(rng.bytes(16)))
            h1 = Host(1, ctxh, DeterministicRNG(rng.bytes(16)))
            alpha = rng.randint(PARAMS.domain)
            beta = rng.randint(PARAMS.modulus)
            delta, _, _ = run_keygen(c0, c1, h0, h1, cnt, alpha, beta, rng)
            rec_h += (vendor_recover_alpha(VENDOR_KEY, ctxh.sid, cnt, delta,
                                           PARAMS) == alpha)
        lo, hi = clopper_pearson(rec_h, Q1_TRIALS)
        run.log(f"  same adversary vs an HONEST chip: {rec_h}/{Q1_TRIALS} "
                f"[{lo:.4f},{hi:.4f}]")
        rows.append({"part": "Q1_recovery", "cell": "honest-control", "p": "", "c": "",
                     "trials": Q1_TRIALS, "value": rec_h / Q1_TRIALS,
                     "lo": lo, "hi": hi, "note": "negative control"})

        # ================================================================ Q2
        run.log("Q2: is the backdoor visible to anything short of a recompute?")
        # (a) do the backdoored keys still compute the right function?
        rng = DeterministicRNG(f"{SEEDS[name]}|q2a")
        ctx2, keys2 = pair_chips(rng, PARAMS_SMALL)
        ok = 0
        for t in range(Q2_KEYS):
            cnt = t
            c0 = VendorBackdoorChip(0, ctx2, keys2[0], VENDOR_KEY, [cnt],
                                    DeterministicRNG(rng.bytes(16)))
            c1 = VendorBackdoorChip(1, ctx2, keys2[1], VENDOR_KEY, [cnt],
                                    DeterministicRNG(rng.bytes(16)))
            c0.gen(cnt); c1.gen(cnt)
            h0 = Host(0, ctx2, DeterministicRNG(rng.bytes(16)))
            h1 = Host(1, ctx2, DeterministicRNG(rng.bytes(16)))
            alpha = rng.randint(PARAMS_SMALL.domain)
            beta = rng.randint(PARAMS_SMALL.modulus)
            d, k0, k1 = run_keygen(c0, c1, h0, h1, cnt, alpha, beta, rng)
            f0, f1 = dpf_eval_full(k0), dpf_eval_full(k1)
            D = PARAMS_SMALL.domain
            ok += all(((f0[(x + d) % D] + f1[(x + d) % D]) % PARAMS_SMALL.modulus)
                      == (beta if x == alpha else 0) for x in range(D))
        lo, hi = clopper_pearson(ok, Q2_KEYS)
        run.log(f"  backdoored keys correct on every domain point: {ok}/{Q2_KEYS} "
                f"[{lo:.4f},{hi:.4f}] -- correctness testing sees nothing")
        rows.append({"part": "Q2_invisibility", "cell": "key-correctness", "p": "",
                     "c": "", "trials": Q2_KEYS, "value": ok / Q2_KEYS,
                     "lo": lo, "hi": hi, "note": "full-domain check"})

        # (b) do the backdoored targets look non-uniform?
        rng = DeterministicRNG(f"{SEEDS[name]}|q2b")
        _, sid = rng.bytes(32), rng.bytes(16)
        root = rng.bytes(32)
        hon = [derive_template_from_slotkey(slot_key(root, j), j, PARAMS).r
               for j in range(Q2_SAMPLES)]
        bad = [vendor_target(VENDOR_KEY, sid, j, PARAMS) for j in range(Q2_SAMPLES)]
        nb = 128
        for tag, xs in (("honest", hon), ("backdoored", bad)):
            counts = [0] * nb
            for v in xs:
                counts[v * nb // PARAMS.domain] += 1
            stat, dof, pval = chi_square_uniform(counts)
            run.log(f"  target uniformity [{tag}]: chi^2={stat:.1f} dof={dof} "
                    f"p={pval:.3f}")
            rows.append({"part": "Q2_invisibility", "cell": f"chi2-{tag}", "p": "",
                         "c": "", "trials": Q2_SAMPLES, "value": stat, "lo": dof,
                         "hi": pval, "note": "value=chi2, lo=dof, hi=p-value"})
        ks, ksp = two_sample_ks(hon, bad)
        run.log(f"  two-sample KS honest vs backdoored targets: D={ks:.5f} "
                f"p={ksp:.3f} ({'indistinguishable' if ksp > 0.05 else 'DISTINGUISHABLE'})")
        rows.append({"part": "Q2_invisibility", "cell": "ks-honest-vs-backdoored",
                     "p": "", "c": "", "trials": Q2_SAMPLES, "value": ks,
                     "lo": "", "hi": ksp, "note": "value=D statistic, hi=p-value"})

        # (c) does the well-formedness check alone catch it?
        rng = DeterministicRNG(f"{SEEDS[name]}|q2c")
        ctx3, keys3 = pair_chips(rng, PARAMS)
        chip = VendorBackdoorChip(0, ctx3, keys3[0], VENDOR_KEY, counters,
                                  DeterministicRNG(rng.bytes(16)))
        chip.gen_batch(counters)
        wf_pass = sum(1 for j in counters
                      if template_is_wellformed(chip.slots[j].template, spot_checks=32))
        run.log(f"  backdoored templates passing the well-formedness check alone: "
                f"{wf_pass}/{len(counters)} -- check (ii) alone is blind to it")
        rows.append({"part": "Q2_invisibility", "cell": "wellformedness-only", "p": "",
                     "c": "", "trials": len(counters),
                     "value": wf_pass / len(counters), "lo": "", "hi": "",
                     "note": "32 spot checks per template"})

        # ================================================================ Q3
        run.log("Q3: does PACT catch it, at the predicted rate?")
        for p in P_GRID:
            for c in C_GRID:
                rng = DeterministicRNG(f"{SEEDS[name]}|q3|{p}|{c}")
                ctx4, keys4 = pair_chips(rng, PARAMS)
                caught = upheld = 0
                for t in range(Q3_TRIALS):
                    plan = blind_taint_choice(counters, c, rng)
                    ch = VendorBackdoorChip(0, ctx4, keys4[0], VENDOR_KEY,
                                            plan.taints, DeterministicRNG(rng.bytes(16)))
                    ch.gen_batch(counters)
                    rep = audit_batch(ch, counters, p, DeterministicRNG(rng.bytes(16)),
                                      DeterministicRNG(rng.bytes(16)))
                    if rep.caught:
                        caught += 1
                        upheld += judgement(rep.certificate, ctx4.attest_pub[0],
                                            PARAMS)[0]
                lo, hi = wilson_ci(caught, Q3_TRIALS)
                ana = 1 - (1 - p) ** c
                run.log(f"  p={p:<5} c={c:<3} caught={caught/Q3_TRIALS:.4f} "
                        f"[{lo:.4f},{hi:.4f}] analytic={ana:.4f} "
                        f"certificates upheld {upheld}/{caught}")
                assert upheld == caught, "a certificate failed public verification"
                rows.append({"part": "Q3_detection", "cell": "vendor-backdoor",
                             "p": p, "c": c, "trials": Q3_TRIALS,
                             "value": caught / Q3_TRIALS, "lo": lo, "hi": hi,
                             "note": f"analytic={ana:.4f}; upheld={upheld}/{caught}"})

        # ================================================================ Q4
        run.log("Q4: how many queries are deanonymised before the first catch?")
        for p in P_GRID:
            rng = DeterministicRNG(f"{SEEDS[name]}|q4|{p}")
            stolen = []
            for t in range(Q4_TRIALS):
                k = 0
                while True:
                    if rng.random() < p:      # this tainted slot was opened
                        break
                    k += 1
                    if k > 100000:
                        break
                stolen.append(k)
            mu, lo, hi = mean_ci(stolen)
            theo_mu, _ = geometric_stats(p)
            med = sorted(stolen)[len(stolen) // 2]
            p95 = sorted(stolen)[int(0.95 * len(stolen))]
            run.log(f"  p={p:<5} mean stolen before first catch = {mu:.2f} "
                    f"[{lo:.2f},{hi:.2f}] (theory {theo_mu:.2f}); "
                    f"median {med}, 95th pct {p95}; "
                    f"P[steal >= 128 undetected] = {(1-p)**128:.3e}")
            rows.append({"part": "Q4_budget", "cell": "stolen-before-catch", "p": p,
                         "c": "", "trials": Q4_TRIALS, "value": mu, "lo": lo,
                         "hi": hi,
                         "note": f"theory={theo_mu:.3f};median={med};p95={p95};"
                                 f"P[>=128]={(1-p)**128:.3e};"
                                 f"P[>=1024]={(1-p)**1024:.3e}"})

        run.finish(rows, summary={
            "recovery_backdoored": recovered / Q1_TRIALS,
            "recovery_honest_control": rec_h / Q1_TRIALS,
            "backdoored_keys_correct": ok / Q2_KEYS,
            "ks_p_value": ksp,
            "wellformedness_pass_rate": wf_pass / len(counters),
            "note": "the attack is invisible to correctness, to distributional "
                    "tests, and to the well-formedness half of the audit; only "
                    "the recompute against a pre-input commitment sees it",
        })


if __name__ == "__main__":
    main()
