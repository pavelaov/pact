from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EXP = os.path.join(HERE, "experiments")
sys.path.insert(0, HERE)

EXPERIMENTS = [
    ("exp1_correctness", "correctness of the protocol and of the audit"),
    ("exp2_detection", "the deterrent factor vs. its closed form"),
    ("exp3_yield", "sacrifice calibration, frontier, provisioning"),
    ("exp4_adaptive", "blind / coin-grinding / oracle arms"),
    ("exp5_defamation", "nine forgery strategies + positive control"),
    ("exp6_consistency", "input consistency arms + ablation"),
    ("exp7_frontier", "the auditability/privacy frontier"),
    ("exp8_auditcost", "assurance cost vs. independent baselines"),
    ("exp9_backdoor", "an implemented backdoor and the leakage budget"),
    ("exp10_costmicro", "measured object sizes, certificates, exposure window"),
    ("exp11_distinguisher", "is the backdoored target stream distinguishable?"),
    ("exp12_singlechip", "cross-chip consistency vs single-chip and colluding backdoors"),
]

FAST_OVERRIDES = {
    "exp1_correctness": {"N_E2E": 50, "N_AUDIT": 400},
    "exp2_detection": {"TRIALS": 500, "XCHECK_TRIALS": 20},
    "exp3_yield": {"TRIALS": 200},
    "exp4_adaptive": {"TRIALS": 300, "PIPE_TRIALS": 20, "GRIND_TRIES": 8},
    "exp5_defamation": {"TRIALS": 40, "CONTROL_TRIALS": 40},
    "exp6_consistency": {"TRIALS": 60},
    "exp7_frontier": {"TRIALS": 20},
    "exp9_backdoor": {"Q1_TRIALS": 50, "Q2_KEYS": 20, "Q2_SAMPLES": 2000,
                      "Q3_TRIALS": 100, "Q4_TRIALS": 500},
    "exp10_costmicro": {"REPS": 20, "BIND_REPS": 50},
    "exp11_distinguisher": {"R": 8, "S": 2000},
}


def selftest() -> bool:
    """A fast end-to-end check that the implementation is sound."""
    from pact.adversary import (VendorBackdoorChip, blind_taint_choice,
                                vendor_recover_alpha)
    from pact.audit import (FORGERY_STRATEGIES, audit_batch, forge_certificate,
                            judgement)
    from pact.dpf import DPFParams, dpf_eval_full
    from pact.primitives import DeterministicRNG, aes_mmo_hash
    from pact.protocol import (CheatProfile, Chip, Host, InputConsistencyAbort,
                               eval_key, pair_chips, run_keygen)

    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    p = DPFParams(n=10, m=32)
    rng = DeterministicRNG(1234)
    ctx, keys = pair_chips(rng, p)
    cnts = list(range(48))
    c0, c1 = Chip(0, ctx, keys[0]), Chip(1, ctx, keys[1])
    c0.gen_batch(cnts); c1.gen_batch(cnts)
    h0 = Host(0, ctx, DeterministicRNG(b"h0"))
    h1 = Host(1, ctx, DeterministicRNG(b"h1"))

    alpha, beta = 613, 4242424
    d, k0, k1 = run_keygen(c0, c1, h0, h1, 3, alpha, beta, DeterministicRNG(7))
    f0, f1 = dpf_eval_full(k0), dpf_eval_full(k1)
    D = p.domain
    check("GEN->BIND->EVAL correct on the full domain",
          all(((f0[(x + d) % D] + f1[(x + d) % D]) % p.modulus)
              == (beta if x == alpha else 0) for x in range(D)))
    check("AES-MMO hash available on the chip datapath",
          len(aes_mmo_hash(b"x")) == 16)

    hb = Host(0, ctx, DeterministicRNG(b"h0"), cheat_bind=True)
    ca, cb = Chip(0, ctx, keys[0]), Chip(1, ctx, keys[1])
    ca.gen_batch(cnts); cb.gen_batch(cnts)
    caught = False
    try:
        run_keygen(ca, cb, hb, h1, 3, alpha, beta, DeterministicRNG(7))
    except InputConsistencyAbort:
        caught = True
    check("malicious host that binds a share it did not submit is caught", caught)

    ch = Chip(0, ctx, keys[0]); ch.gen_batch(cnts)
    rep = audit_batch(ch, cnts, 0.2, DeterministicRNG(b"a"), DeterministicRNG(b"b"))
    check("honest chip is never accused", not rep.caught)

    cc = Chip(0, ctx, keys[0], cheat=CheatProfile(frozenset(cnts), "seed"),
              rng=DeterministicRNG(9))
    cc.gen_batch(cnts)
    rep = audit_batch(cc, cnts, 0.5, DeterministicRNG(b"a"), DeterministicRNG(b"b"))
    check("cheating chip is caught", rep.caught)
    check("certificate verifies from public data alone",
          rep.caught and judgement(rep.certificate, ctx.attest_pub[0], p)[0])
    check("certificate does not verify against the wrong key",
          rep.caught and not judgement(rep.certificate, ctx.attest_pub[1], p)[0])

    all_rejected = True
    for strat in FORGERY_STRATEGIES:
        fresh = Chip(0, ctx, keys[0]); fresh.gen_batch(cnts)
        cert = forge_certificate(fresh, cnts, 0.2, DeterministicRNG(b"f0"),
                                 DeterministicRNG(b"f1"), strat)
        all_rejected &= not judgement(cert, ctx.attest_pub[0], p)[0]
    check(f"all {len(FORGERY_STRATEGIES)} forgery strategies rejected", all_rejected)

    vk = bytes(range(32))
    b0 = VendorBackdoorChip(0, ctx, keys[0], vk, [5], DeterministicRNG(1))
    b1 = VendorBackdoorChip(1, ctx, keys[1], vk, [5], DeterministicRNG(2))
    b0.gen(5); b1.gen(5)
    d, k0, k1 = run_keygen(b0, b1, h0, h1, 5, alpha, beta, DeterministicRNG(11))
    check("backdoor recovers the secret index from public data",
          vendor_recover_alpha(vk, ctx.sid, 5, d, p) == alpha)
    f0, f1 = dpf_eval_full(k0), dpf_eval_full(k1)
    check("backdoored key is still perfectly correct",
          all(((f0[(x + d) % D] + f1[(x + d) % D]) % p.modulus)
              == (beta if x == alpha else 0) for x in range(D)))
    return ok


def run_one(name: str, fast: bool) -> float:
    t0 = time.time()
    if fast and name in FAST_OVERRIDES:
        code = ("import sys; sys.path.insert(0, r'%s');\n"
                "import %s as E;\n" % (EXP, name))
        for k, v in FAST_OVERRIDES[name].items():
            code += f"E.{k} = {v}\n"
        code += "E.main()\n"
        cmd = [sys.executable, "-c", code]
    else:
        cmd = [sys.executable, os.path.join(EXP, name + ".py")]
    r = subprocess.run(cmd, cwd=EXP, capture_output=True, text=True)
    dt = time.time() - t0
    if r.returncode != 0:
        print(r.stdout[-3000:])
        print(r.stderr[-3000:])
        raise SystemExit(f"{name} failed with exit code {r.returncode}")
    return dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--only", choices=[name for name, _ in EXPERIMENTS])
    args = ap.parse_args()
    if not selftest():
        raise SystemExit(1)
    if args.check:
        return
    names = [args.only] if args.only else [name for name, _ in EXPERIMENTS]
    for name in names:
        print(f"{name}: {run_one(name, args.fast):.2f}s")

if __name__ == "__main__":
    main()
