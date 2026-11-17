#!/usr/bin/env python3
"""
E7 (NEW) -- The auditability/privacy frontier: is auditability decided by
*where* generation does its work?

Research question.  PACT's thesis is that whether a trusted FSS key generator
can be audited *without exposing inputs* is decided by where it does its
input-independent work.  That is a claim about alternatives, so we implement
the alternatives and measure both sides of the trade-off on all of them under
an identical workload.

Falsifiable hypothesis.
  H1  For every placement other than PACT's, audit coverage and input privacy
      are in strict 1:1 conflict: the residual entropy of a delivered key's
      secret target falls to (1-q)*n bits when a q-fraction of generation
      events is audited, reaching 0 as coverage approaches 1.
  H2  PACT holds residual entropy at the full n bits for every audit rate, and
      pays instead in *throughput*: 1/(1-q) generation events per delivered
      key.
H1 is falsified if any counterfactual placement keeps positive residual
entropy at coverage 1; H2 is falsified if PACT's measured residual entropy
ever falls below n bits, or if a recovery adversary beats random guessing.

Workload (identical for every arm).  Each trial must *deliver* K usable keys.
An arm generates as many slots as it needs to do that; slots consumed by the
audit count against its throughput, not against its privacy.  This removes the
degenerate comparison in which one arm simply produces fewer keys.

Measured quantities, per (placement, q, n) cell:
  coverage        P[the audit catches a covert generator that taints c events]
  alpha-recovery  fraction of *delivered* keys whose secret target a concrete
                  adversary -- reading every public value and every audit
                  opening -- recovers exactly
  residual bits   mean surviving entropy of a delivered key's target
  gen-per-key     generation events consumed per delivered key
all with Wilson 95% intervals on the proportions.

Two extra controls:
  * a stronger cross-slot adversary that pairs every opened target with every
    public shift, to check PACT's zero-recovery is not an artefact of a weak
    adversary;
  * a chi-square test that the public shift Delta is uniform, i.e. that it
    carries no information about the target on its own.
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import Run, SEEDS
from pact.dpf import DPFParams
from pact.placements import (Generator, Placement, cross_slot_guess,
                             recover_inputs)
from pact.primitives import DeterministicRNG
from pact.stats import chi_square_uniform, wilson_ci

K_DELIVER = 64                     # usable keys each trial must deliver
TRIALS = 600
Q_GRID = (0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90)
C_TAINT = 4                        # generation events a covert chip taints
N_GRID = (8, 12, 16, 20)
PLACEMENTS = (Placement.MONOLITHIC, Placement.LATE_COMMIT,
              Placement.NO_SACRIFICE, Placement.PACT)
MAX_EVENTS = 4000                  # safety bound on one trial


def run_cell(placement: Placement, q: float, params: DPFParams,
             trials: int, rng: DeterministicRNG):
    caught_batches = tainted_batches = 0
    delivered = recovered = 0
    residual_bits = 0.0
    events_total = 0
    cross_hits = cross_total = 0
    deltas = []

    sacrifices_possible = placement in (Placement.PACT,)

    # A covert generator taints C_TAINT of the events it actually performs.
    # PACT performs more events per delivered key (sacrificed slots), so the
    # taint pool is sized to the events that will occur -- otherwise the arms
    # would not be compared at matched adversary effort.
    n_events_expected = (int(round(K_DELIVER / max(1e-9, 1 - q)))
                         if sacrifices_possible else K_DELIVER)

    for _ in range(trials):
        root, sid = rng.bytes(32), rng.bytes(16)
        taint = set()
        pool = list(range(n_events_expected))
        for _ in range(min(C_TAINT, len(pool))):
            taint.add(pool.pop(rng.randint(len(pool))))

        gen = Generator(placement, root, sid, params, taint=taint,
                        rng=DeterministicRNG(rng.bytes(16)))
        openings = {}
        n_delivered = 0
        j = 0
        caught = False
        while n_delivered < K_DELIVER and j < MAX_EVENTS:
            gen.gen(j)
            marked = rng.random() < q
            if placement is Placement.PACT:
                # committed before any input exists -> open first, sacrifice
                if marked:
                    openings[j] = gen.open(j)
                else:
                    alpha = rng.randint(params.domain)
                    beta = rng.randint(params.modulus)
                    d = gen.bind(j, alpha, beta)
                    deltas.append(d)
                    n_delivered += 1
            else:
                # nothing openable exists until the input has been consumed
                alpha = rng.randint(params.domain)
                beta = rng.randint(params.modulus)
                d = gen.bind(j, alpha, beta)
                if d is not None:
                    deltas.append(d)
                n_delivered += 1
                if marked and gen.openable(j):
                    openings[j] = gen.open(j)
            j += 1
        events_total += j

        # --- the audit ------------------------------------------------------
        for jj, op in openings.items():
            if gen.audit_check(jj, op) is not None:
                caught = True
        if taint:
            tainted_batches += 1
            caught_batches += caught

        # --- the input-recovery adversary -----------------------------------
        for jj, g in recover_inputs(gen, openings, params).items():
            delivered += 1
            if g is not None and g == gen.slots[jj].alpha:
                recovered += 1
            else:
                residual_bits += params.n
        for jj, g in cross_slot_guess(gen, openings, params).items():
            cross_total += 1
            cross_hits += int(g == gen.slots[jj].alpha)

    cov_lo, cov_hi = wilson_ci(caught_batches, max(tainted_batches, 1))
    rec_lo, rec_hi = wilson_ci(recovered, max(delivered, 1))
    cr_lo, cr_hi = wilson_ci(cross_hits, max(cross_total, 1))
    return {
        "coverage": caught_batches / max(tainted_batches, 1),
        "coverage_lo": cov_lo, "coverage_hi": cov_hi,
        "alpha_recovery": recovered / max(delivered, 1),
        "recovery_lo": rec_lo, "recovery_hi": rec_hi,
        "residual_bits": residual_bits / max(delivered, 1),
        "residual_frac": residual_bits / max(delivered, 1) / params.n,
        "gen_per_key": events_total / max(delivered, 1),
        "delivered": delivered,
        "cross_slot_recovery": cross_hits / max(cross_total, 1),
        "cross_lo": cr_lo, "cross_hi": cr_hi,
        "deltas": deltas,
    }


def main():
    name = "exp7_frontier"
    with Run(name, {"K_deliver": K_DELIVER, "trials": TRIALS,
                    "q_grid": list(Q_GRID), "c_taint": C_TAINT,
                    "n_grid": list(N_GRID),
                    "placements": [p.value for p in PLACEMENTS]}) as run:
        rows = []
        n_ref = max(N_GRID)
        run.log("H1: every placement but PACT trades coverage against privacy 1:1.")
        run.log("H2: PACT holds privacy and pays in throughput instead.")
        for n in N_GRID:
            params = DPFParams(n=n, m=32)
            for placement in PLACEMENTS:
                for q in Q_GRID:
                    rng = DeterministicRNG(f"{SEEDS[name]}|{n}|{placement.value}|{q}")
                    r = run_cell(placement, q, params, TRIALS, rng)
                    rows.append({
                        "n": n, "placement": placement.value, "q": q,
                        "trials": TRIALS, "c_taint": C_TAINT,
                        "coverage": round(r["coverage"], 5),
                        "coverage_lo": round(r["coverage_lo"], 5),
                        "coverage_hi": round(r["coverage_hi"], 5),
                        "analytic_coverage": round(1 - (1 - q) ** C_TAINT, 5),
                        "alpha_recovery": round(r["alpha_recovery"], 5),
                        "recovery_lo": round(r["recovery_lo"], 5),
                        "recovery_hi": round(r["recovery_hi"], 5),
                        "residual_bits": round(r["residual_bits"], 4),
                        "residual_frac": round(r["residual_frac"], 5),
                        "gen_per_key": round(r["gen_per_key"], 4),
                        "delivered": r["delivered"],
                        "cross_slot_recovery": round(r["cross_slot_recovery"], 6),
                        "cross_hi": round(r["cross_hi"], 6),
                    })
                    if n == n_ref:
                        run.log(f"n={n} {placement.value:<13} q={q:<5} "
                                f"cov={r['coverage']:.3f} "
                                f"rec={r['alpha_recovery']:.3f} "
                                f"resid={r['residual_bits']:.2f}/{n}b "
                                f"gen/key={r['gen_per_key']:.3f}")

        # ---- verdict -------------------------------------------------------
        hi_q = max(Q_GRID)
        at_hi = {r["placement"]: r for r in rows
                 if r["q"] == hi_q and r["n"] == n_ref}
        run.log(f"--- verdict at n={n_ref}, q={hi_q}")
        for pl, r in at_hi.items():
            run.log(f"  {pl:<13} coverage={r['coverage']:.3f} "
                    f"residual={r['residual_bits']:.2f}/{n_ref} bits "
                    f"recovery={r['alpha_recovery']:.3f} "
                    f"gen/key={r['gen_per_key']:.3f}")
        pact = at_hi[Placement.PACT.value]
        others = [r for k, r in at_hi.items() if k != Placement.PACT.value]
        h1 = all(abs(o["residual_frac"] - (1 - hi_q)) < 0.05 for o in others)
        h2 = (abs(pact["residual_bits"] - n_ref) < 1e-9
              and pact["alpha_recovery"] == 0.0
              and abs(pact["gen_per_key"] - 1 / (1 - hi_q)) < 0.15)
        run.log(f"H1 supported (counterfactual arms trace 1-q): {h1}")
        run.log(f"H2 supported (PACT holds n bits, pays 1/(1-q) events/key): {h2}")

        # ---- sensitivity: does the *opening strategy* matter? --------------
        # A deterministic audit schedule is predictable, so a covert chip can
        # place every taint outside it.  This is the same lesson as the
        # commit-then-challenge ordering, one level up.
        run.log("--- opening-strategy sensitivity at n=%d (q=0.25, c=%d)"
                % (n_ref, C_TAINT))
        pref = DPFParams(n=n_ref, m=32)
        for strategy in ("public-coin", "prefix", "suffix"):
            for chip_knows in (False, True):
                rng = DeterministicRNG(f"{SEEDS[name]}|strat|{strategy}|{chip_knows}")
                caught = trials_s = 0
                for _ in range(TRIALS):
                    q = 0.25
                    n_events = int(round(K_DELIVER / (1 - q)))
                    schedule = set()
                    if strategy == "public-coin":
                        schedule = {j for j in range(n_events) if rng.random() < q}
                    elif strategy == "prefix":
                        schedule = set(range(int(q * n_events)))
                    else:
                        schedule = set(range(n_events - int(q * n_events), n_events))
                    if chip_knows:
                        safe = [j for j in range(n_events) if j not in schedule]
                        taints = set(safe[:C_TAINT])
                    else:
                        pool = list(range(n_events))
                        taints = set()
                        for _ in range(C_TAINT):
                            taints.add(pool.pop(rng.randint(len(pool))))
                    trials_s += 1
                    caught += int(bool(taints & schedule))
                lo, hi = wilson_ci(caught, trials_s)
                rows.append({"n": n_ref, "placement": f"strategy:{strategy}",
                             "q": 0.25, "trials": trials_s, "c_taint": C_TAINT,
                             "coverage": round(caught / trials_s, 5),
                             "coverage_lo": round(lo, 5), "coverage_hi": round(hi, 5),
                             "analytic_coverage": round(1 - 0.75 ** C_TAINT, 5),
                             "alpha_recovery": "", "recovery_lo": "", "recovery_hi": "",
                             "residual_bits": "", "residual_frac": "",
                             "gen_per_key": round(1 / 0.75, 4), "delivered": "",
                             "cross_slot_recovery": "",
                             "cross_hi": "chip_knows_schedule=%d" % int(chip_knows)})
                run.log(f"  {strategy:<12} chip_knows_schedule={int(chip_knows)} "
                        f"coverage={caught/trials_s:.4f} [{lo:.4f},{hi:.4f}]")

        # ---- is the public shift uniform on its own? -----------------------
        params = DPFParams(n=12, m=32)
        rng = DeterministicRNG(f"{SEEDS[name]}|delta")
        r = run_cell(Placement.PACT, 0.10, params, 200, rng)
        nb = 64
        counts = [0] * nb
        for d in r["deltas"]:
            counts[d * nb // params.domain] += 1
        stat, dof, pval = chi_square_uniform(counts)
        run.log(f"public shift Delta over {len(r['deltas'])} binds: "
                f"chi^2={stat:.1f} dof={dof} p={pval:.3f} "
                f"({'consistent with uniform' if pval > 0.01 else 'NOT uniform'})")

        run.finish(rows, summary={
            "H1_supported": bool(h1), "H2_supported": bool(h2),
            "q_max": hi_q, "n_ref": n_ref,
            "pact_residual_bits": pact["residual_bits"],
            "pact_recovery": pact["alpha_recovery"],
            "pact_cross_slot_recovery_upper95": pact["cross_hi"],
            "pact_gen_per_key": pact["gen_per_key"],
            "delta_chi2": round(stat, 3), "delta_chi2_dof": dof,
            "delta_chi2_p": round(pval, 5),
            "delta_samples": len(r["deltas"]),
        })


if __name__ == "__main__":
    main()
