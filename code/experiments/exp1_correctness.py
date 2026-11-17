#!/usr/bin/env python3
"""
E1 -- Correctness of the protocol and of the audit.

Three questions:
  (a) does GEN -> BIND -> EVAL compute the intended FSS functionality?
  (b) does an honestly generated slot always pass the audit (no false
      accusation)?
  (c) is a deviating slot always caught *when it is opened*?

(b) and (c) are the 20,000-trial Monte-Carlo pair the paper reports; (a) is a
full-domain check on every one of a few thousand random keys, which is the
strongest correctness statement available at this domain size.
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import Run, SEEDS
from pact.audit import audit_batch, check_opened_slot, coin_toss, judgement
from pact.dpf import (DPFParams, derive_template_from_slotkey, dpf_eval_full,
                      tamper_template)
from pact.primitives import (DeterministicRNG, h, slot_commit, slot_key,
                             slot_key_commit)
from pact.protocol import (CheatProfile, Chip, Host, InputConsistencyAbort,
                           eval_key, pair_chips, run_keygen)
from pact.stats import clopper_pearson, rule_of_three, wilson_ci

import struct

N_E2E = 2000          # end-to-end keys, each checked over its FULL domain
N_AUDIT = 20000       # honest slots, and again tampered slots
TAMPER_MODES = ("seed", "target", "cw", "fcw")
P_E2E = DPFParams(n=12, m=32)      # full-domain check is 4096 points per key
P_AUD = DPFParams(n=16, m=32)      # audit at deployment domain size


def main():
    name = "exp1_correctness"
    with Run(name, {"n_e2e": N_E2E, "n_audit": N_AUDIT,
                    "params_e2e": f"n={P_E2E.n},m={P_E2E.m}",
                    "params_audit": f"n={P_AUD.n},m={P_AUD.m}",
                    "tamper_modes": list(TAMPER_MODES)}) as run:
        rng = DeterministicRNG(SEEDS[name])
        rows = []

        # ------------------------------------------------------------------
        # (a) end-to-end functional correctness, checked over the full domain
        # ------------------------------------------------------------------
        ctx, keys = pair_chips(rng, P_E2E)
        c0, c1 = Chip(0, ctx, keys[0]), Chip(1, ctx, keys[1])
        h0 = Host(0, ctx, DeterministicRNG(b"h0|" + struct.pack(">Q", SEEDS[name])))
        h1 = Host(1, ctx, DeterministicRNG(b"h1|" + struct.pack(">Q", SEEDS[name])))
        cnts = list(range(N_E2E))
        c0.gen_batch(cnts); c1.gen_batch(cnts)

        ok_e2e = 0
        for i in cnts:
            alpha = rng.randint(P_E2E.domain)
            beta = rng.randint(P_E2E.modulus)
            delta, k0, k1 = run_keygen(c0, c1, h0, h1, i, alpha, beta, rng)
            f0, f1 = dpf_eval_full(k0), dpf_eval_full(k1)
            D = P_E2E.domain
            good = all(((f0[(x + delta) % D] + f1[(x + delta) % D]) % P_E2E.modulus)
                       == (beta if x == alpha else 0) for x in range(D))
            ok_e2e += good
        run.log(f"(a) end-to-end correctness: {ok_e2e}/{N_E2E} keys correct on all "
                f"{P_E2E.domain} domain points "
                f"({ok_e2e * P_E2E.domain} point checks)")
        lo, hi = clopper_pearson(ok_e2e, N_E2E)
        rows.append({"part": "e2e_correctness", "cell": f"n={P_E2E.n}", "trials": N_E2E,
                     "successes": ok_e2e, "rate": ok_e2e / N_E2E,
                     "ci_lo": lo, "ci_hi": hi, "note": "full-domain check per key"})

        # ------------------------------------------------------------------
        # (b) honest slots must always pass  (defamation freeness, empirical)
        # ------------------------------------------------------------------
        ctxA, keysA = pair_chips(DeterministicRNG(SEEDS[name] + 1), P_AUD)
        passed = 0
        for cnt in range(N_AUDIT):
            rho_cnt = slot_key(ctxA.root, cnt)
            tmpl = derive_template_from_slotkey(rho_cnt, cnt, P_AUD)
            nonce = h(b"PACT|slotnonce", rho_cnt, struct.pack(">Q", cnt))[:16]
            digest = h(b"PACT|slotdigest", struct.pack(">Q", cnt), tmpl.witness_bytes())
            gamma = slot_commit(ctxA.sid, cnt, digest, nonce)
            delta = slot_key_commit(rho_cnt)
            reason = check_opened_slot(ctxA.sid, cnt, tmpl.witness_bytes(), nonce,
                                       rho_cnt, gamma, delta, P_AUD, spot_checks=8)
            passed += (reason is None)
        false_acc = N_AUDIT - passed
        run.log(f"(b) honest slots passing audit: {passed}/{N_AUDIT}; "
                f"false accusations {false_acc} "
                f"(95% upper bound on false-accusation rate {rule_of_three(N_AUDIT):.2e})")
        lo, hi = clopper_pearson(passed, N_AUDIT)
        rows.append({"part": "honest_pass", "cell": "all", "trials": N_AUDIT,
                     "successes": passed, "rate": passed / N_AUDIT,
                     "ci_lo": lo, "ci_hi": hi,
                     "note": f"false-accusation rate <= {rule_of_three(N_AUDIT):.2e} (95%)"})

        # ------------------------------------------------------------------
        # (c) a deviating slot, once opened, must always be caught
        # ------------------------------------------------------------------
        per_mode = N_AUDIT // len(TAMPER_MODES)
        caught_total = 0
        for mi, mode in enumerate(TAMPER_MODES):
            trng = DeterministicRNG(b"tamper|" + mode.encode())
            caught = 0
            reasons = {}
            for j in range(per_mode):
                cnt = mi * per_mode + j
                rho_cnt = slot_key(ctxA.root, cnt)
                honest = derive_template_from_slotkey(rho_cnt, cnt, P_AUD)
                bad = tamper_template(honest, trng, mode)
                nonce = h(b"PACT|slotnonce", rho_cnt, struct.pack(">Q", cnt))[:16]
                # the chip commits to what it actually emitted
                digest = h(b"PACT|slotdigest", struct.pack(">Q", cnt), bad.witness_bytes())
                gamma = slot_commit(ctxA.sid, cnt, digest, nonce)
                dcom = slot_key_commit(rho_cnt)
                reason = check_opened_slot(ctxA.sid, cnt, bad.witness_bytes(), nonce,
                                           rho_cnt, gamma, dcom, P_AUD, spot_checks=8)
                caught += (reason is not None)
                reasons[reason] = reasons.get(reason, 0) + 1
            caught_total += caught
            lo, hi = clopper_pearson(caught, per_mode)
            run.log(f"(c) tampered[{mode}]: caught {caught}/{per_mode}  reasons={reasons}")
            rows.append({"part": "tampered_caught", "cell": mode, "trials": per_mode,
                         "successes": caught, "rate": caught / per_mode,
                         "ci_lo": lo, "ci_hi": hi,
                         "note": ";".join(f"{k}={v}" for k, v in reasons.items())})
        lo, hi = clopper_pearson(caught_total, per_mode * len(TAMPER_MODES))
        rows.append({"part": "tampered_caught", "cell": "all",
                     "trials": per_mode * len(TAMPER_MODES), "successes": caught_total,
                     "rate": caught_total / (per_mode * len(TAMPER_MODES)),
                     "ci_lo": lo, "ci_hi": hi, "note": "pooled over modes"})
        run.log(f"(c) pooled: {caught_total}/{per_mode*len(TAMPER_MODES)} opened "
                f"deviations caught; miss rate upper bound "
                f"{rule_of_three(per_mode*len(TAMPER_MODES)):.2e} (95%)")

        run.finish(rows, summary={
            "e2e_correct": ok_e2e, "e2e_trials": N_E2E,
            "honest_pass": passed, "honest_trials": N_AUDIT,
            "false_accusations": false_acc,
            "tampered_caught": caught_total,
            "tampered_trials": per_mode * len(TAMPER_MODES),
            "miss_rate_upper_95": rule_of_three(per_mode * len(TAMPER_MODES)),
            "false_acc_upper_95": rule_of_three(N_AUDIT),
        })


if __name__ == "__main__":
    main()
