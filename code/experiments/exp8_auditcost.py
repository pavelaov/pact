#!/usr/bin/env python3
"""
E8 -- Publicly verifiable assurance cost per usable key.

The frontier experiment shows PACT can audit without exposing inputs.  This
experiment places PACT on a cost-per-deterrence axis against independent
routes to evidence that a DPF key generator behaved honestly.

PACT's side is MEASURED from the running implementation: the audit witness,
the certificate, the coin transcript and the per-key amortisation are all read
out of real objects.  The baselines are ANALYTIC models built from published
parameters of the cited works, with every parameter named, a sensitivity band
where the literature gives a range, and an explicit incomparability note where
the security notion differs.  We do not claim to beat schemes that provide a
strictly stronger notion; we quantify what each notion costs.

Baselines (all independent of the hardware substrate PACT builds on):
  attestation-only        remote attestation of firmware identity.  No per-key
                          evidence, no transferable certificate: deterrence 0.
  trusted dealer          origin of the axis: no evidence at all.
  PVC-compiled 2PC gen    the closest notion match: covert with public
                          verifiability, deterrence 1 - 1/lambda from
                          lambda-fold replication of a dealerless generator.
  cut-and-choose 2PC      full malicious security at 2^-s from s copies.
  actively secure 2P-FSS  always-on certification of every key.
  publicly verifiable ZK  a NIZK that the emitted key is the honest expansion
                          of a committed target: deterministic assurance.

The dependent variable is publicly verifiable audit traffic per USABLE key at
a target deterrence epsilon, plus rounds added on the key-delivery path and
whether the evidence is transferable.
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import Run, SEEDS
from pact.audit import audit_batch, coin_toss, judgement
from pact.dpf import DPFParams
from pact.primitives import DeterministicRNG
from pact.protocol import CheatProfile, Chip, pair_chips

EPS_GRID = (0.05, 0.10, 0.20, 0.50, 0.90, 0.99)
N_GRID = (8, 12, 16, 20, 24)
BATCH = 256
LAMBDA_BITS = 128

# ---------------------------------------------------------------------------
# Analytic baseline models.  Every constant is named and sourced in the paper.
# ---------------------------------------------------------------------------

# Dealerless two-party DPF generation, per key, modelled from the structure of
# the protocol (n sequential rounds; per level two seeds plus a correction word
# exchanged under OT).  Reported as a band because the source states
# asymptotics, not byte counts.
def dealerless_base_bytes(n: int, lo=True) -> float:
    per_level_bits = (4 * LAMBDA_BITS + (LAMBDA_BITS + 2))
    b = n * per_level_bits / 8.0
    return b * (1.0 if lo else 2.0)


def pvc_lambda_for(eps: float) -> float:
    """Replication factor giving covert deterrence eps = 1 - 1/lambda."""
    return 1.0 / max(1e-9, 1.0 - eps)


def cutchoose_s_for(eps: float) -> float:
    """Copies for statistical malicious security; eps here is 1 - 2^-s."""
    if eps >= 1.0:
        return float("inf")
    return max(1.0, -math.log2(max(1e-12, 1.0 - eps)))


# Actively secure two-party FSS with cross-phase certification: always-on
# certification traffic per instance (constant rounds).
ACTIVE_CERT_BYTES = 1282.0

# Publicly verifiable ZK of correct generation (VOLE-in-the-head family).
# A DPF Gen at depth n costs ~4n AES-128 evaluations; the marginal publicly
# verifiable proof rate per AES is swept over a band.
ZK_BYTES_PER_AES = (500.0, 2000.0, 5900.0)


def zk_bytes(n: int, rate: float) -> float:
    return 4 * n * rate


# Publicly verifiable covert compiler: constant-size certificate at the
# 128-bit level.
PVC_CERT_BYTES = 354.0


def main():
    name = "exp8_auditcost"
    with Run(name, {"eps_grid": list(EPS_GRID), "n_grid": list(N_GRID),
                    "batch": BATCH, "lambda_bits": LAMBDA_BITS,
                    "active_cert_bytes": ACTIVE_CERT_BYTES,
                    "zk_bytes_per_aes": list(ZK_BYTES_PER_AES),
                    "pvc_cert_bytes": PVC_CERT_BYTES}) as run:
        rows = []

        # ------------------------------------------------------------------
        # (1) MEASURE PACT's own audit objects from the implementation
        # ------------------------------------------------------------------
        measured = {}
        for n in N_GRID:
            params = DPFParams(n=n, m=32)
            rng = DeterministicRNG(f"{SEEDS[name]}|{n}")
            ctx, keys = pair_chips(rng, params)
            counters = list(range(BATCH))
            chip = Chip(0, ctx, keys[0])
            chip.gen_batch(counters)
            slot = chip.slots[0]
            witness = len(slot.template.witness_bytes())
            keyshare = slot.template.key_share(0, 0).size_bytes
            root, sig, tree = chip.sign_batch(counters)
            coin = coin_toss(root, DeterministicRNG(b"c0"), DeterministicRNG(b"c1"))
            open_bytes = witness + len(slot.nonce) + len(slot.slot_key)
            batch_bytes = len(root) + len(sig) + coin.bytes_on_wire()
            # a real certificate, produced by a real catch
            cheat = Chip(0, ctx, keys[0], cheat=CheatProfile(frozenset(counters), "seed"),
                         rng=DeterministicRNG(b"x"))
            cheat.gen_batch(counters)
            rep = audit_batch(cheat, counters, 0.02, DeterministicRNG(b"r0"),
                              DeterministicRNG(b"r1"))
            cert_bytes = rep.certificate.size_bytes() if rep.certificate else 0
            guilty = judgement(rep.certificate, ctx.attest_pub[0], params)[0] if rep.certificate else False
            measured[n] = {"witness": witness, "open": open_bytes,
                           "keyshare": keyshare, "batch": batch_bytes,
                           "cert": cert_bytes, "cert_upheld": guilty}
            run.log(f"n={n:<3} measured: key share {keyshare} B, audit witness "
                    f"{witness} B, full opening {open_bytes} B, "
                    f"batch overhead {batch_bytes} B/{BATCH} slots, "
                    f"certificate {cert_bytes} B (upheld={guilty})")
            rows.append({"part": "pact-measured", "n": n, "eps": "",
                         "scheme": "pact", "bytes_per_usable_key": "",
                         "rounds_added": 0, "publicly_verifiable": 1,
                         "notion": "covert, deterrence p",
                         "key_share_bytes": keyshare, "witness_bytes": witness,
                         "opening_bytes": open_bytes, "cert_bytes": cert_bytes,
                         "note": f"batch overhead {batch_bytes} B per {BATCH} slots"})

        # ------------------------------------------------------------------
        # (2) COST PER USABLE KEY AT MATCHED DETERRENCE
        # ------------------------------------------------------------------
        n_ref = 16
        m = measured[n_ref]
        base_lo = dealerless_base_bytes(n_ref, lo=True)
        base_hi = dealerless_base_bytes(n_ref, lo=False)
        run.log(f"--- cost per usable key at n={n_ref}; dealerless base "
                f"{base_lo:.0f}-{base_hi:.0f} B/key")

        for eps in EPS_GRID:
            # PACT: audit rate p = eps for a single tainted event; an audited
            # slot is sacrificed, so amortised traffic per USABLE key is
            # p/(1-p) * opening bytes, plus the batch overhead.
            p = eps
            pact_bytes = (p / (1 - p)) * m["open"] + m["batch"] / BATCH / (1 - p)
            rows.append({"part": "frontier", "n": n_ref, "eps": eps,
                         "scheme": "PACT", "bytes_per_usable_key": round(pact_bytes, 2),
                         "rounds_added": 0, "publicly_verifiable": 1,
                         "notion": "publicly verifiable covert, deterrence p",
                         "key_share_bytes": m["keyshare"], "witness_bytes": m["witness"],
                         "opening_bytes": m["open"], "cert_bytes": m["cert"],
                         "note": "measured; audit is off the key-delivery path"})

            lam = pvc_lambda_for(eps)
            for tag, base in (("lo", base_lo), ("hi", base_hi)):
                rows.append({"part": "frontier", "n": n_ref, "eps": eps,
                             "scheme": f"PVC-compiled dealerless 2PC ({tag})",
                             "bytes_per_usable_key": round(lam * base, 2),
                             "rounds_added": n_ref, "publicly_verifiable": 1,
                             "notion": "publicly verifiable covert, deterrence 1-1/lambda",
                             "key_share_bytes": m["keyshare"], "witness_bytes": "",
                             "opening_bytes": "", "cert_bytes": PVC_CERT_BYTES,
                             "note": f"lambda={lam:.2f}; analytic, base band {tag}"})

            s = cutchoose_s_for(eps)
            if math.isfinite(s):
                rows.append({"part": "frontier", "n": n_ref, "eps": eps,
                             "scheme": "cut-and-choose malicious 2PC",
                             "bytes_per_usable_key": round(s * base_lo, 2),
                             "rounds_added": n_ref, "publicly_verifiable": 0,
                             "notion": "malicious, soundness 2^-s (NOT a covert deterrent)",
                             "key_share_bytes": m["keyshare"], "witness_bytes": "",
                             "opening_bytes": "", "cert_bytes": "",
                             "note": f"s={s:.1f} copies; incomparable notion"})

            rows.append({"part": "frontier", "n": n_ref, "eps": eps,
                         "scheme": "actively secure 2P-FSS certification",
                         "bytes_per_usable_key": ACTIVE_CERT_BYTES,
                         "rounds_added": "O(1)", "publicly_verifiable": 0,
                         "notion": "active security (stronger; always on)",
                         "key_share_bytes": "", "witness_bytes": "",
                         "opening_bytes": "", "cert_bytes": "",
                         "note": "constant in eps: certification is not tunable"})

            for rate in ZK_BYTES_PER_AES:
                rows.append({"part": "frontier", "n": n_ref, "eps": eps,
                             "scheme": f"publicly verifiable ZK of generation ({rate:.0f} B/AES)",
                             "bytes_per_usable_key": round(zk_bytes(n_ref, rate), 2),
                             "rounds_added": 0, "publicly_verifiable": 1,
                             "notion": "deterministic assurance (stronger)",
                             "key_share_bytes": "", "witness_bytes": "",
                             "opening_bytes": "", "cert_bytes": "",
                             "note": f"{4*n_ref} AES-128 evaluations proved"})

            for scheme in ("attestation-only trusted hardware", "trusted dealer"):
                rows.append({"part": "frontier", "n": n_ref, "eps": eps,
                             "scheme": scheme, "bytes_per_usable_key": 0.0,
                             "rounds_added": 0, "publicly_verifiable": 0,
                             "notion": "no per-key evidence; deterrence 0",
                             "key_share_bytes": "", "witness_bytes": "",
                             "opening_bytes": "", "cert_bytes": "",
                             "note": "plotted at eps=0; shown at every eps as a floor"})

            run.log(f"eps={eps:<5} PACT {pact_bytes:9.1f} B/key | "
                    f"PVC {lam*base_lo:10.1f}-{lam*base_hi:.1f} | "
                    f"active {ACTIVE_CERT_BYTES:.0f} | "
                    f"ZK {zk_bytes(n_ref, ZK_BYTES_PER_AES[0]):.0f}-"
                    f"{zk_bytes(n_ref, ZK_BYTES_PER_AES[-1]):.0f}")

        # ------------------------------------------------------------------
        # (3) how the frontier scales with the domain size
        # ------------------------------------------------------------------
        run.log("--- scaling with domain size at eps = 0.10")
        for n in N_GRID:
            mm = measured[n]
            p = 0.10
            pact_bytes = (p / (1 - p)) * mm["open"] + mm["batch"] / BATCH / (1 - p)
            rows.append({"part": "scaling", "n": n, "eps": p, "scheme": "PACT",
                         "bytes_per_usable_key": round(pact_bytes, 2),
                         "rounds_added": 0, "publicly_verifiable": 1,
                         "notion": "publicly verifiable covert",
                         "key_share_bytes": mm["keyshare"],
                         "witness_bytes": mm["witness"], "opening_bytes": mm["open"],
                         "cert_bytes": mm["cert"], "note": "measured"})
            for tag, lo in (("lo", True), ("hi", False)):
                b = dealerless_base_bytes(n, lo=lo)
                rows.append({"part": "scaling", "n": n, "eps": p,
                             "scheme": f"PVC-compiled dealerless 2PC ({tag})",
                             "bytes_per_usable_key": round(pvc_lambda_for(p) * b, 2),
                             "rounds_added": n, "publicly_verifiable": 1,
                             "notion": "publicly verifiable covert",
                             "key_share_bytes": "", "witness_bytes": "",
                             "opening_bytes": "", "cert_bytes": PVC_CERT_BYTES,
                             "note": "analytic"})
            rows.append({"part": "scaling", "n": n, "eps": p,
                         "scheme": "publicly verifiable ZK of generation (2000 B/AES)",
                         "bytes_per_usable_key": round(zk_bytes(n, 2000.0), 2),
                         "rounds_added": 0, "publicly_verifiable": 1,
                         "notion": "deterministic assurance",
                         "key_share_bytes": "", "witness_bytes": "",
                         "opening_bytes": "", "cert_bytes": "", "note": "analytic"})
            run.log(f"n={n:<3} PACT {pact_bytes:8.1f} B/key   "
                    f"ZK {zk_bytes(n,2000.0):9.0f}   "
                    f"PVC {pvc_lambda_for(p)*dealerless_base_bytes(n):9.1f}")

        m16 = measured[16]
        p = 0.10
        pact10 = (p / (1 - p)) * m16["open"] + m16["batch"] / BATCH / (1 - p)
        run.finish(rows, summary={
            "pact_bytes_per_key_at_eps_0.10_n16": round(pact10, 2),
            "pact_certificate_bytes_n16": m16["cert"],
            "pact_witness_bytes_n16": m16["witness"],
            "pact_key_share_bytes_n16": m16["keyshare"],
            "ratio_active_over_pact": round(ACTIVE_CERT_BYTES / pact10, 1),
            "ratio_zk_mid_over_pact": round(zk_bytes(16, 2000.0) / pact10, 1),
            "note": "baselines are analytic models from published parameters; "
                    "notions differ and the incomparability is stated per row",
        })


if __name__ == "__main__":
    main()
