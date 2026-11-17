"""
Alternative *placements* of FSS key generation, implemented for real.

The paper's thesis is that whether a trusted key generator can be audited is
decided by *where* it does its work.  A thesis of that shape can only be
tested against the alternatives, so this module implements the counterfactual
placements as working generators with working audits, using the same
primitives, the same commitment machinery and the same third-party judge as
PACT.

    MONOLITHIC          one-shot generation.  The chip reconstructs the input
                        and expands the whole key in a single pass.  The only
                        object an auditor can open is the derivation of a
                        *real* key -- whose target IS the secret input.

    LATE_COMMIT         the input-independent work is done offline, but the
                        commitment is published only at bind time, so the
                        object that gets opened is the slot that was bound.

    NO_SACRIFICE        the commitment is published before any input exists,
                        but audited slots are returned to the pool and later
                        bound instead of being sacrificed.

    PACT                commitment before any input exists, and an audited
                        slot is sacrificed.

Every placement exposes the same interface, so the experiment harness can
sweep them uniformly:

    gen(j)                      -> GenRecord (public commitment + private state)
    bind(j, alpha, beta)        -> public Delta  (None if the placement binds
                                   at generation time, i.e. MONOLITHIC)
    openable(j)                 -> True if slot j may still be opened
    open(j)                     -> Opening (slot_key, nonce, witness)
    audit_check(j, opening)     -> failure reason or None
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from .dpf import (DPFParams, Template, derive_template_from_slotkey,
                  gen_template_from_randomness, tamper_template)
from .primitives import (DeterministicRNG, LAMBDA_BYTES, h, kdf, slot_commit,
                         slot_key, slot_key_commit)


class Placement(Enum):
    MONOLITHIC = "monolithic"
    LATE_COMMIT = "late-commit"
    NO_SACRIFICE = "no-sacrifice"
    PACT = "pact"


@dataclass
class SlotRec:
    j: int
    template: Optional[Template] = None
    slot_key: bytes = b""
    nonce: bytes = b""
    gamma: bytes = b""          # slot commitment (empty until published)
    dcom: bytes = b""           # H(slot_key)
    committed_at: str = ""      # "gen" | "bind"
    bound: bool = False
    sacrificed: bool = False
    delta: Optional[int] = None  # the PUBLIC shift emitted at bind
    alpha: Optional[int] = None  # ground truth, for scoring only
    tainted: bool = False


@dataclass
class Opening:
    j: int
    slot_key: bytes
    nonce: bytes
    witness: bytes


class Generator:
    """A key generator at one placement."""

    def __init__(self, placement: Placement, root: bytes, sid: bytes,
                 params: DPFParams, taint: Optional[set] = None,
                 rng: Optional[DeterministicRNG] = None, tamper_mode: str = "seed"):
        self.placement = placement
        self.root = root
        self.sid = sid
        self.params = params
        self.taint = set(taint or ())
        self.rng = rng or DeterministicRNG(b"gen")
        self.tamper_mode = tamper_mode
        self.slots: Dict[int, SlotRec] = {}

    # ---------------------------------------------------------------- GEN
    def gen(self, j: int) -> SlotRec:
        rec = SlotRec(j=j)
        rec.slot_key = slot_key(self.root, j)
        rec.dcom = slot_key_commit(rec.slot_key)
        rec.nonce = kdf(rec.slot_key, b"PACT|slotnonce", j, 16)
        if self.placement is Placement.MONOLITHIC:
            # nothing can happen before the input arrives: generation IS bind
            self.slots[j] = rec
            return rec
        tmpl = derive_template_from_slotkey(rec.slot_key, j, self.params)
        if j in self.taint:
            tmpl = tamper_template(tmpl, self.rng, self.tamper_mode)
            rec.tainted = True
        rec.template = tmpl
        if self.placement in (Placement.PACT, Placement.NO_SACRIFICE):
            rec.gamma = self._commit(j, tmpl, rec.nonce)
            rec.committed_at = "gen"
        self.slots[j] = rec
        return rec

    def _commit(self, j: int, tmpl: Template, nonce: bytes) -> bytes:
        digest = h(b"PACT|slotdigest", struct.pack(">Q", j), tmpl.witness_bytes())
        return slot_commit(self.sid, j, digest, nonce)

    # --------------------------------------------------------------- BIND
    def bind(self, j: int, alpha: int, beta: int) -> Optional[int]:
        rec = self.slots[j]
        p = self.params
        if self.placement is Placement.MONOLITHIC:
            # one pass: reconstruct the input and expand the whole key for it
            material = kdf(rec.slot_key, b"PACT|onepass", j, 2 * LAMBDA_BYTES)
            s0_0 = material[:LAMBDA_BYTES]
            s0_1 = material[LAMBDA_BYTES:2 * LAMBDA_BYTES]
            tmpl = gen_template_from_randomness(j, alpha, s0_0, s0_1, p)
            if j in self.taint:
                tmpl = tamper_template(tmpl, self.rng, self.tamper_mode)
                rec.tainted = True
            rec.template = tmpl
            rec.gamma = self._commit(j, tmpl, rec.nonce)
            rec.committed_at = "bind"
            rec.bound = True
            rec.alpha = alpha
            rec.delta = None          # there is no shift: the key targets alpha
            return None
        if rec.sacrificed:
            raise RuntimeError("slot already sacrificed")
        if self.placement is Placement.LATE_COMMIT:
            rec.gamma = self._commit(j, rec.template, rec.nonce)
            rec.committed_at = "bind"
        rec.bound = True
        rec.alpha = alpha
        rec.delta = (rec.template.r - alpha) % p.domain
        return rec.delta

    # -------------------------------------------------------------- AUDIT
    def openable(self, j: int) -> bool:
        rec = self.slots[j]
        if self.placement is Placement.MONOLITHIC:
            return rec.bound          # only a real key exists to open
        if self.placement is Placement.LATE_COMMIT:
            return rec.bound          # nothing is committed before the bind
        return not rec.sacrificed     # PACT / NO_SACRIFICE: templates exist early

    def open(self, j: int) -> Opening:
        rec = self.slots[j]
        if self.placement is Placement.PACT:
            rec.sacrificed = True     # the defining move: opened => never used
        return Opening(j=j, slot_key=rec.slot_key, nonce=rec.nonce,
                       witness=rec.template.witness_bytes())

    def audit_check(self, j: int, op: Opening) -> Optional[str]:
        """Recompute the opened object against its commitment."""
        rec = self.slots[j]
        if slot_key_commit(op.slot_key) != rec.dcom:
            return "slotkey-mismatch"
        digest = h(b"PACT|slotdigest", struct.pack(">Q", j), op.witness)
        if slot_commit(self.sid, j, digest, op.nonce) != rec.gamma:
            return "commitment-mismatch"
        if self.placement is Placement.MONOLITHIC:
            # the auditor can only recompute the expansion once it knows the
            # target, which the witness supplies -- and that is the leak
            material = kdf(op.slot_key, b"PACT|onepass", j, 2 * LAMBDA_BYTES)
            from .audit import parse_witness
            parsed = parse_witness(op.witness, self.params)
            if parsed is None:
                return "malformed-template"
            honest = gen_template_from_randomness(
                j, parsed.r, material[:LAMBDA_BYTES],
                material[LAMBDA_BYTES:2 * LAMBDA_BYTES], self.params)
            if honest.witness_bytes() != op.witness:
                return "derivation-mismatch"
            return None
        honest = derive_template_from_slotkey(op.slot_key, j, self.params)
        if honest.witness_bytes() != op.witness:
            return "derivation-mismatch"
        return None


# ---------------------------------------------------------------------------
# The input-recovery adversary that scores each placement's privacy
# ---------------------------------------------------------------------------


def recover_inputs(gen: Generator, openings: Dict[int, Opening],
                   params: DPFParams) -> Dict[int, Optional[int]]:
    """Best-effort recovery of the secret target of every *bound* slot.

    The adversary sees:
      * every public commitment,
      * every public shift Delta emitted at bind,
      * every audit opening (which is public by construction -- the audit is
        publicly verifiable, so the openings are published),
    and nothing else.  It returns its guess for each bound slot's alpha, or
    ``None`` where it has no better strategy than guessing uniformly.
    """
    from .audit import parse_witness

    out: Dict[int, Optional[int]] = {}
    # targets revealed by openings, indexed by the slot they belong to
    revealed: Dict[int, int] = {}
    for j, op in openings.items():
        t = parse_witness(op.witness, params)
        if t is not None:
            revealed[j] = t.r

    for j, rec in gen.slots.items():
        if not rec.bound:
            continue
        if gen.placement is Placement.MONOLITHIC:
            # the opened witness IS a key for alpha: reading r gives alpha
            out[j] = revealed.get(j)
        else:
            if j in revealed and rec.delta is not None:
                out[j] = (revealed[j] - rec.delta) % params.domain
            else:
                out[j] = None
    return out


def cross_slot_guess(gen: Generator, openings: Dict[int, Opening],
                     params: DPFParams) -> Dict[int, int]:
    """A stronger adversary for PACT: try every opened target against every
    public shift.  It has no reason to work -- the opened slots and the bound
    slots are different objects with independent targets -- and the experiment
    measures that it does not."""
    from .audit import parse_witness
    revealed = []
    for op in openings.values():
        t = parse_witness(op.witness, params)
        if t is not None:
            revealed.append(t.r)
    guesses: Dict[int, int] = {}
    for j, rec in gen.slots.items():
        if rec.bound and rec.delta is not None and revealed:
            # pick the first cross-pairing; any fixed rule is equally arbitrary
            guesses[j] = (revealed[0] - rec.delta) % params.domain
    return guesses
