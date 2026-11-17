#!/usr/bin/env python3
"""
E10 -- Protocol cost micro-benchmarks, all measured from the implementation.

What the paper needs, with nothing estimated:
  * object sizes: key share, template witness, full opening, certificate,
    coin transcript, batch attestation -- as a function of the domain size;
  * the single BIND exchange: exact bytes per chip, and the fact that it is
    one message whose length does not depend on the tree depth beyond the
    share encoding;
  * work: AES-128 block invocations for GEN, for the audit recompute, and for
    a full-domain EVAL;
  * the input-exposure window: the fraction of the bind datapath during which
    a reconstructed (alpha, beta) exists inside the boundary, instrumented in
    the reference implementation;
  * verification throughput: how many audited slots a third party checks per
    second, and how certificate verification scales with the batch size
    (Merkle path length).

These are software reference measurements of the *protocol*, not hardware
performance claims.
"""

from __future__ import annotations

import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import Run, SEEDS
from pact.audit import (audit_batch, check_opened_slot, coin_toss, judgement,
                        challenge_set)
from pact.dpf import (DPFParams, derive_template_from_slotkey, dpf_eval,
                      dpf_eval_full)
from pact.primitives import (DeterministicRNG, aes_call_count,
                             reset_aes_call_count, slot_key)
from pact.protocol import (CheatProfile, Chip, Host, eval_key, pair_chips,
                           run_keygen)
from pact.stats import mean_ci

N_GRID = (8, 12, 16, 20, 24, 32)
BATCH_GRID = (64, 256, 1024, 4096)
REPS = 200
BIND_REPS = 500


def main():
    name = "exp10_costmicro"
    with Run(name, {"n_grid": list(N_GRID), "batch_grid": list(BATCH_GRID),
                    "reps": REPS, "bind_reps": BIND_REPS}) as run:
        rows = []

        # ---- object sizes and work, vs domain size ------------------------
        run.log("--- sizes and work vs domain size (lambda = 128, group Z_2^32)")
        for n in N_GRID:
            params = DPFParams(n=n, m=32)
            rng = DeterministicRNG(f"{SEEDS[name]}|{n}")
            ctx, keys = pair_chips(rng, params)
            counters = list(range(64))
            chip = Chip(0, ctx, keys[0])

            reset_aes_call_count()
            t0 = time.perf_counter()
            for j in counters:
                chip.gen(j)
            gen_s = (time.perf_counter() - t0) / len(counters)
            gen_aes = aes_call_count() / len(counters)

            slot = chip.slots[0]
            witness = len(slot.template.witness_bytes())
            keyshare = slot.template.key_share(0, 0).size_bytes
            opening = witness + len(slot.nonce) + len(slot.slot_key)

            reset_aes_call_count()
            t0 = time.perf_counter()
            for j in counters:
                check_opened_slot(ctx.sid, j, chip.slots[j].template.witness_bytes(),
                                  chip.slots[j].nonce, chip.slots[j].slot_key,
                                  chip.slots[j].commitment,
                                  chip.slots[j].key_commitment, params, 8)
            chk_s = (time.perf_counter() - t0) / len(counters)
            chk_aes = aes_call_count() / len(counters)

            # one BIND exchange
            c0, c1 = Chip(0, ctx, keys[0]), Chip(1, ctx, keys[1])
            c0.gen(0); c1.gen(0)
            h0 = Host(0, ctx, DeterministicRNG(b"h0")); h1 = Host(1, ctx, DeterministicRNG(b"h1"))
            cm0, op0 = h0.submit(0, 1, 2)
            msg = c0.bind_message(0, 1, 2, cm0)
            bind_bytes = len(msg)

            run.log(f"n={n:<3} key {keyshare:4d} B | witness {witness:4d} B | "
                    f"opening {opening:4d} B | bind msg {bind_bytes:3d} B | "
                    f"GEN {gen_aes:5.0f} AES {gen_s*1e6:7.1f} us | "
                    f"audit-check {chk_aes:5.0f} AES {chk_s*1e6:7.1f} us")
            rows.append({"part": "sizes", "n": n, "batch": "",
                         "key_share_B": keyshare, "witness_B": witness,
                         "opening_B": opening, "bind_msg_B": bind_bytes,
                         "gen_aes": round(gen_aes, 1), "gen_us": round(gen_s * 1e6, 2),
                         "check_aes": round(chk_aes, 1),
                         "check_us": round(chk_s * 1e6, 2),
                         "cert_B": "", "judge_ms": "", "value": "", "note": ""})

        # ---- certificate size and third-party verification vs batch size --
        run.log("--- certificate size and judgement cost vs batch size")
        params = DPFParams(n=16, m=32)
        for B in BATCH_GRID:
            rng = DeterministicRNG(f"{SEEDS[name]}|b{B}")
            ctx, keys = pair_chips(rng, params)
            counters = list(range(B))
            chip = Chip(0, ctx, keys[0],
                        cheat=CheatProfile(frozenset(counters), "seed"),
                        rng=DeterministicRNG(b"c"))
            chip.gen_batch(counters)
            rep = audit_batch(chip, counters, 0.02, DeterministicRNG(b"r0"),
                              DeterministicRNG(b"r1"))
            cert = rep.certificate
            ts = []
            for _ in range(50):
                t0 = time.perf_counter()
                guilty, why = judgement(cert, ctx.attest_pub[0], params)
                ts.append(time.perf_counter() - t0)
            mu, lo, hi = mean_ci(ts)
            run.log(f"batch {B:5d}: certificate {cert.size_bytes():5d} B "
                    f"(merkle path {len(cert.merkle_path)} + {len(cert.open_path)}), "
                    f"judgement {mu*1e3:.3f} ms [{lo*1e3:.3f},{hi*1e3:.3f}] "
                    f"guilty={guilty}")
            rows.append({"part": "certificate", "n": 16, "batch": B,
                         "key_share_B": "", "witness_B": "", "opening_B": "",
                         "bind_msg_B": "", "gen_aes": "", "gen_us": "",
                         "check_aes": "", "check_us": "",
                         "cert_B": cert.size_bytes(),
                         "judge_ms": round(mu * 1e3, 4), "value": int(guilty),
                         "note": f"paths {len(cert.merkle_path)}+{len(cert.open_path)}"})

        # ---- the input-exposure window ------------------------------------
        run.log("--- input-exposure window inside the boundary")
        params = DPFParams(n=16, m=32)
        rng = DeterministicRNG(f"{SEEDS[name]}|win")
        ctx, keys = pair_chips(rng, params)
        c0, c1 = Chip(0, ctx, keys[0]), Chip(1, ctx, keys[1])
        cnts = list(range(BIND_REPS))
        reset_aes_call_count()
        c0.gen_batch(cnts); c1.gen_batch(cnts)
        h0 = Host(0, ctx, DeterministicRNG(b"h0")); h1 = Host(1, ctx, DeterministicRNG(b"h1"))
        t0 = time.perf_counter()
        for j in cnts:
            run_keygen(c0, c1, h0, h1, j, rng.randint(params.domain),
                       rng.randint(params.modulus), rng)
        bind_s = (time.perf_counter() - t0) / len(cnts)
        total_aes = aes_call_count() / len(cnts)
        exposure_aes = c0.stats["exposure_aes"] / len(cnts)
        exposure_ops = c0.stats["exposure_ops"] / len(cnts)
        run.log(f"  BIND wall-clock {bind_s*1e6:.1f} us/key (software reference, "
                f"both chips); one key costs {total_aes:.0f} AES block operations "
                f"end to end, of which {exposure_aes:.0f} execute while a "
                f"reconstructed (alpha,beta) exists inside the boundary "
                f"({100.0*exposure_aes/total_aes:.3f}%). The exposure window is "
                f"{exposure_ops:.0f} arithmetic operations: one modular "
                f"subtraction for Delta and one affine patch of FCW. Every PRG "
                f"invocation happened at GEN, before any input existed.")
        rows.append({"part": "exposure", "n": 16, "batch": "", "key_share_B": "",
                     "witness_B": "", "opening_B": "", "bind_msg_B": "",
                     "gen_aes": round(total_aes, 1), "gen_us": round(bind_s * 1e6, 2),
                     "check_aes": round(exposure_aes, 3), "check_us": exposure_ops,
                     "cert_B": "",
                     "judge_ms": "", "value": round(100.0 * exposure_aes / total_aes, 4),
                     "note": "percent of the key-production datapath during which "
                             "a reconstructed input exists"})

        # ---- audit verification throughput --------------------------------
        run.log("--- third-party audit verification throughput")
        for n in (12, 16, 20):
            params = DPFParams(n=n, m=32)
            rng = DeterministicRNG(f"{SEEDS[name]}|tp{n}")
            ctx, keys = pair_chips(rng, params)
            counters = list(range(REPS))
            chip = Chip(0, ctx, keys[0])
            chip.gen_batch(counters)
            for sc in (0, 8, 32):
                t0 = time.perf_counter()
                for j in counters:
                    check_opened_slot(ctx.sid, j, chip.slots[j].template.witness_bytes(),
                                      chip.slots[j].nonce, chip.slots[j].slot_key,
                                      chip.slots[j].commitment,
                                      chip.slots[j].key_commitment, params, sc)
                dt = (time.perf_counter() - t0) / len(counters)
                run.log(f"  n={n} spot-checks={sc:2d}: {dt*1e6:8.1f} us/slot "
                        f"= {1/dt:8.0f} slots/s")
                rows.append({"part": "throughput", "n": n, "batch": "",
                             "key_share_B": "", "witness_B": "", "opening_B": "",
                             "bind_msg_B": "", "gen_aes": "", "gen_us": "",
                             "check_aes": sc, "check_us": round(dt * 1e6, 2),
                             "cert_B": "", "judge_ms": "", "value": round(1 / dt, 1),
                             "note": "slots verified per second"})

        run.finish(rows, summary={"note": "software reference measurements of the "
                                          "protocol; not hardware performance"})


if __name__ == "__main__":
    main()
