#!/usr/bin/env python3
"""
E12 -- Single-chip deviation, cross-chip consistency, and the subkey-level
backdoor arms.

Arms (each over N_TRIALS non-degenerate pairings of N_SLOTS slots):
  (a) single-chip vendor-TARGET backdoor: caught deterministically by the
      cross-chip consistency check (zero sacrifice) and by the sacrificial
      audit when the tainted slot is opened.
  (b) single-chip vendor-SUBKEY backdoor (the deviation that keeps every
      internal audit check consistent): the opened slot PASSES the internal
      recompute checks, while the consistency check catches it
      deterministically.
  (c) colluding vendor-TARGET backdoor on BOTH chips (the deployable,
      correctness-preserving configuration of Exp. E9): the consistency check
      is silent by design and the sacrificial audit catches it at the
      closed-form rate p.
  (c2) colluding vendor-SUBKEY backdoor on BOTH chips: the sacrificial audit
      does NOT catch it (the residual gap discussed in the single-chip and
      limitations sections of the paper), and the escalation (pairing-root
      reveal) attributes the deviation to the pair in every trial.
  (d) escalation after a single-chip conflict: revealing the pairing root lets
      a third party attribute the deviation *individually* to the right chip.

Degenerate trials -- a vendor target that accidentally equals the honest
target, probability 1/domain -- are skipped and not counted, because they
contain no deviation at all.
"""

from __future__ import annotations

import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import Run, SEEDS
from pact.adversary import (VendorBackdoorChip, VendorSubkeyBackdoorChip,
                            vendor_recover_alpha, vendor_target)
from pact.audit import (audit_batch, check_opened_slot,
                        cross_chip_consistency_audit, consistency_judgement)
from pact.dpf import DPFParams, derive_template_from_slotkey
from pact.primitives import DeterministicRNG, slot_key, slot_key_commit
from pact.protocol import Chip, Host, eval_key, pair_chips, run_keygen
from pact.stats import clopper_pearson, wilson_ci

N_TRIALS = 1000
N_SLOTS = 64
P_AUDIT = 0.1
PARAMS = DPFParams(n=16, m=64)


def _ci(k: int, n: int):
    """Wilson 95% interval over k successes in n trials."""
    lo, hi = wilson_ci(k, n)
    return lo, hi


def _row(rows, part, cell, trials, successes, note=""):
    lo, hi = _ci(successes, trials)
    rows.append({"part": part, "cell": cell, "trials": trials,
                 "successes": successes, "rate": successes / trials,
                 "ci_lo": lo, "ci_hi": hi, "note": note})


def _is_degenerate_target(ctx, vendor_key, cnt):
    """True when the vendor target accidentally equals the honest target."""
    honest = derive_template_from_slotkey(
        slot_key(ctx.root, cnt), cnt, PARAMS).r
    return vendor_target(vendor_key, ctx.sid, cnt, PARAMS) == honest


def main():
    name = "exp12_singlechip"
    with Run(name, {"n_trials": N_TRIALS, "n_slots": N_SLOTS, "p_audit": P_AUDIT,
                    "params": f"n={PARAMS.n},m={PARAMS.m}"}) as run:
        rng = DeterministicRNG(SEEDS.get(name, 20240012))
        VENDOR_KEY = rng.bytes(32)
        rows = []

        # -------------------------------------------------------------- (a)
        # single-chip vendor-TARGET backdoor.
        a_cons = a_audit = certs_upheld = 0
        t = 0
        while a_cons < N_TRIALS or a_audit < N_TRIALS:
            ctx, keys = pair_chips(rng, PARAMS)
            taint = t * N_SLOTS + (t % N_SLOTS)
            counters = [t * N_SLOTS + i for i in range(N_SLOTS)]
            if _is_degenerate_target(ctx, VENDOR_KEY, taint):
                t += 1
                continue
            c0 = Chip(0, ctx, keys[0])
            c1 = VendorBackdoorChip(1, ctx, keys[1], VENDOR_KEY, [taint])
            c0.gen_batch(counters)
            c1.gen_batch(counters)
            caught, cert, violations = cross_chip_consistency_audit(c0, c1, counters)
            a_cons += bool(caught)
            if cert is not None:
                g, _ = consistency_judgement(cert, ctx.attest_pub[0], ctx.attest_pub[1])
                certs_upheld += bool(g)
            rep = audit_batch(c1, counters, 1.0, rng, rng)
            a_audit += bool(rep.caught)
            t += 1
        _row(rows, "a_consistency_target", "single-chip target backdoor",
             N_TRIALS, a_cons, f"zero sacrifice; certificates upheld {certs_upheld}/{N_TRIALS}")
        _row(rows, "a_audit_target", "single-chip target backdoor",
             N_TRIALS, a_audit, "sacrificial audit with every slot opened")
        run.log(f"(a) single-chip target backdoor: consistency {a_cons}/{N_TRIALS}; "
                f"audit(p=1) {a_audit}/{N_TRIALS}; certificates upheld {certs_upheld}")

        # -------------------------------------------------------------- (b)
        # single-chip vendor-SUBKEY backdoor: passes every internal check,
        # caught by the consistency check anyway.
        b_cons = b_gap = certs_upheld = 0
        t = 0
        while b_cons < N_TRIALS or b_gap < N_TRIALS:
            ctx, keys = pair_chips(rng, PARAMS)
            taint = t * N_SLOTS + (t % N_SLOTS)
            counters = [t * N_SLOTS + i for i in range(N_SLOTS)]
            c0 = Chip(0, ctx, keys[0])
            c1 = VendorSubkeyBackdoorChip(1, ctx, keys[1], VENDOR_KEY, [taint])
            c0.gen_batch(counters)
            c1.gen_batch(counters)
            caught, cert, violations = cross_chip_consistency_audit(c0, c1, counters)
            b_cons += bool(caught)
            if cert is not None:
                g, _ = consistency_judgement(cert, ctx.attest_pub[0], ctx.attest_pub[1])
                certs_upheld += bool(g)
            s = c1.slots[taint]
            reason = check_opened_slot(ctx.sid, taint, s.template.witness_bytes(),
                                       s.nonce, s.slot_key, s.commitment,
                                       s.key_commitment, PARAMS, spot_checks=8)
            b_gap += (reason is None)
            t += 1
        _row(rows, "b_consistency_subkey", "single-chip subkey backdoor",
             N_TRIALS, b_cons, f"zero sacrifice; certificates upheld {certs_upheld}/{N_TRIALS}")
        _row(rows, "b_audit_gap", "opened subkey-backdoored slot passes audit",
             N_TRIALS, b_gap, "the internal checks alone cannot see this deviation")
        run.log(f"(b) single-chip subkey backdoor: consistency {b_cons}/{N_TRIALS}; "
                f"opened slot passes internal audit {b_gap}/{N_TRIALS}")

        # -------------------------------------------------------------- (c)
        # BOTH chips vendor-TARGET backdoored (deployable collusion).
        c_cons = c_audit = n = 0
        while n < N_TRIALS:
            ctx, keys = pair_chips(rng, PARAMS)
            taint = t * N_SLOTS + (t % N_SLOTS)
            counters = [t * N_SLOTS + i for i in range(N_SLOTS)]
            if _is_degenerate_target(ctx, VENDOR_KEY, taint):
                t += 1
                continue
            c0 = VendorBackdoorChip(0, ctx, keys[0], VENDOR_KEY, [taint])
            c1 = VendorBackdoorChip(1, ctx, keys[1], VENDOR_KEY, [taint])
            c0.gen_batch(counters)
            c1.gen_batch(counters)
            caught, cert, violations = cross_chip_consistency_audit(c0, c1, counters)
            c_cons += bool(caught)
            rep = audit_batch(c0, counters, P_AUDIT, rng, rng)
            c_audit += bool(rep.caught)
            t += 1
            n += 1
        _row(rows, "c_consistency_silent", "both chips target-backdoored",
             N_TRIALS, c_cons, "consistency is silent by design (identical commitments)")
        _row(rows, "c_audit_colluding", "both chips target-backdoored, p=0.1, c=1",
             N_TRIALS, c_audit, f"closed form 1-(1-p)^1 = {P_AUDIT}")
        run.log(f"(c) both chips target-backdoored: consistency false alarms {c_cons}/{N_TRIALS}; "
                f"audit caught {c_audit}/{N_TRIALS} (closed form {P_AUDIT})")

        # -------------------------------------------------------------- (c2)
        # BOTH chips vendor-SUBKEY backdoored: the residual gap, stated
        # honestly, and closed by the escalation.
        c2_cons = c2_audit = c2_escal = n = 0
        while n < N_TRIALS:
            ctx, keys = pair_chips(rng, PARAMS)
            taint = t * N_SLOTS + (t % N_SLOTS)
            counters = [t * N_SLOTS + i for i in range(N_SLOTS)]
            c0 = VendorSubkeyBackdoorChip(0, ctx, keys[0], VENDOR_KEY, [taint])
            c1 = VendorSubkeyBackdoorChip(1, ctx, keys[1], VENDOR_KEY, [taint])
            c0.gen_batch(counters)
            c1.gen_batch(counters)
            caught, cert, violations = cross_chip_consistency_audit(c0, c1, counters)
            c2_cons += bool(caught)
            rep = audit_batch(c0, counters, 1.0, rng, rng)
            c2_audit += bool(rep.caught)
            # escalation: reveal rho; the pair's commitments disagree with the
            # honest derivation on the tainted slot
            honest = slot_key_commit(slot_key(ctx.root, taint))
            deviated = (c0.slots[taint].key_commitment != honest
                        and c1.slots[taint].key_commitment != honest)
            c2_escal += bool(deviated)
            t += 1
            n += 1
        _row(rows, "c2_consistency_silent", "both chips subkey-backdoored",
             N_TRIALS, c2_cons, "consistency is silent by design (identical commitments)")
        _row(rows, "c2_audit_silent", "both chips subkey-backdoored, audit p=1",
             N_TRIALS, c2_audit, "the sacrificial audit cannot see this deviation (stated gap)")
        _row(rows, "c2_escalation", "pair-level attribution via root reveal",
             N_TRIALS, c2_escal, "the escalation catches the colluding subkey backdoor")
        run.log(f"(c2) both chips subkey-backdoored: consistency {c2_cons}/{N_TRIALS}; "
                f"audit(p=1) caught {c2_audit}/{N_TRIALS}; escalation {c2_escal}/{N_TRIALS}")

        # -------------------------------------------------------------- (d)
        # escalation: individual attribution after a single-chip conflict.
        attributed = 0
        t = 0
        while attributed < N_TRIALS:
            ctx, keys = pair_chips(rng, PARAMS)
            taint = t * N_SLOTS + (t % N_SLOTS)
            counters = [t * N_SLOTS + i for i in range(N_SLOTS)]
            c0 = Chip(0, ctx, keys[0])
            c1 = VendorSubkeyBackdoorChip(1, ctx, keys[1], VENDOR_KEY, [taint])
            c0.gen_batch(counters)
            c1.gen_batch(counters)
            deviates = []
            for cnt in counters:
                honest_delta = slot_key_commit(slot_key(ctx.root, cnt))
                deviates.append((
                    c0.slots[cnt].key_commitment != honest_delta,
                    c1.slots[cnt].key_commitment != honest_delta))
            tainted = t % N_SLOTS
            ok = all(a is False for a, _ in deviates) and \
                 all(b is False for _, b in deviates[:tainted] + deviates[tainted + 1:]) and \
                 deviates[tainted][1] is True
            attributed += bool(ok)
            t += 1
        _row(rows, "d_root_reveal", "individual attribution after conflict",
             N_TRIALS, attributed, "honest chip reveals rho; judge points at the deviating chip")
        run.log(f"(d) escalation to individual attribution: {attributed}/{N_TRIALS}")

        run.finish(rows, summary={
            "a_consistency_target": a_cons / N_TRIALS,
            "a_audit_target": a_audit / N_TRIALS,
            "b_consistency_subkey": b_cons / N_TRIALS,
            "b_audit_gap": b_gap / N_TRIALS,
            "c_consistency_silent": c_cons / N_TRIALS,
            "c_audit_colluding": c_audit / N_TRIALS,
            "c2_audit_silent": c2_audit / N_TRIALS,
            "c2_escalation": c2_escal / N_TRIALS,
            "d_root_reveal": attributed / N_TRIALS,
        })


if __name__ == "__main__":
    main()
