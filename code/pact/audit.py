"""
PACT reference implementation -- the publicly auditable cut-and-choose.

Four mechanisms live here:

  1. ``coin_toss``     commit-then-challenge public coin.  Each host commits to
                       a random string *after* the chip has published and
                       signed its batch root, then both open; the combination
                       seeds the audit challenge.  The ordering is what turns
                       an adaptive chip into a blind one.

  2. ``audit_batch``   sacrificial cut-and-choose.  Each READY slot is marked
                       independently with probability p; a marked slot is
                       opened instead of used.  Opening reveals
                       (rho_cnt, nonce, witness); the auditor recomputes
                       Derive(rho_cnt, cnt) and re-checks the DPF invariant.

  3. ``Certificate``   accountability.  A failed check yields a *transferable*
                       certificate that a third party verifies from PUBLIC
                       data only: the accused chip's attested public key plus
                       the certificate.  This is (Commit, Blame, Judgement) of
                       publicly verifiable covert security.

  4. ``forge_certificate`` / ``judgement``
                       defamation freeness.  Because both the batch
                       commitments *and* the audit-round opening are signed by
                       the chip, no host-side manipulation of an honest chip's
                       transcript yields a certificate that Judgement accepts.

Everything is instrumented (bytes on the wire, verification time) so the
paper's cost numbers come out of the same run that produces its security
numbers.
"""

from __future__ import annotations

import struct
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .dpf import (
    DPFParams,
    Template,
    derive_template_from_slotkey,
    template_is_wellformed,
)
from .primitives import (
    DeterministicRNG,
    LAMBDA_BYTES,
    attest_verify,
    h,
    merkle_verify,
    slot_commit,
    slot_key_commit,
)
from .protocol import (
    Chip,
    SlotState,
    batch_leaf,
    batch_message,
    opening_leaf,
    opening_message,
)


# ---------------------------------------------------------------------------
# 1.  Commit-then-challenge coin toss
# ---------------------------------------------------------------------------


@dataclass
class CoinTranscript:
    batch_root: bytes
    commit0: bytes
    commit1: bytes
    open0: bytes
    open1: bytes
    nonce0: bytes
    nonce1: bytes
    chi: bytes

    def bytes_on_wire(self) -> int:
        return (len(self.commit0) + len(self.commit1)
                + len(self.open0) + len(self.open1)
                + len(self.nonce0) + len(self.nonce1))

    def verify(self) -> bool:
        if h(b"PACT|coin", self.batch_root, self.open0, self.nonce0) != self.commit0:
            return False
        if h(b"PACT|coin", self.batch_root, self.open1, self.nonce1) != self.commit1:
            return False
        return self.chi == h(b"PACT|chi", self.batch_root, self.open0, self.open1)


def coin_toss(batch_root: bytes, rng0: DeterministicRNG, rng1: DeterministicRNG
              ) -> CoinTranscript:
    """Both hosts commit *after* the batch root exists, then both open.

    Neither host controls chi: the commitments are exchanged first, so each
    host's contribution is fixed before it learns the other's.
    """
    x0, nu0 = rng0.bytes(32), rng0.bytes(16)
    x1, nu1 = rng1.bytes(32), rng1.bytes(16)
    c0 = h(b"PACT|coin", batch_root, x0, nu0)
    c1 = h(b"PACT|coin", batch_root, x1, nu1)
    # ... both commitments are published, and only then do both open ...
    chi = h(b"PACT|chi", batch_root, x0, x1)
    return CoinTranscript(batch_root, c0, c1, x0, x1, nu0, nu1, chi)


def slot_is_marked(chi: bytes, index: int, cnt: int, p: float) -> bool:
    d = h(b"PACT|mark", chi, struct.pack(">I", index), struct.pack(">Q", cnt))
    return int.from_bytes(d[:8], "big") / 2.0 ** 64 < p


def challenge_set(chi: bytes, counters: Sequence[int], p: float) -> List[int]:
    """Expand the public coin into per-slot independent Bernoulli(p) marks.

    Deterministic given (chi, counters, p): any third party recomputes the
    audit set from the transcript, which is what makes the audit *public*.
    """
    return [c for i, c in enumerate(counters) if slot_is_marked(chi, i, c, p)]


# ---------------------------------------------------------------------------
# 2 + 3.  Sacrificial audit, blame, judgement
# ---------------------------------------------------------------------------


@dataclass
class Certificate:
    """A transferable certificate of cheating.

    Everything a third party needs, and nothing else:
      * the batch root, signed by the accused chip's attested key, plus the
        Merkle path to the accused slot's leaf (gamma_cnt, delta_cnt);
      * the coin transcript, proving the slot was legitimately challenged;
      * the audit-round opening root, also signed by the accused chip, plus
        the Merkle path to this slot's opening;
      * the opening itself (rho_cnt, nonce, witness).

    The witness and rho_cnt belong to a slot that was *sacrificed and never
    used*, so publishing them exposes no input and no live key.
    """
    sid: bytes
    party: int
    cnt: int
    batch_index: int
    batch_root: bytes
    batch_sig: bytes
    batch_counters_first: int
    batch_size: int
    commitment: bytes          # gamma_cnt
    key_commitment: bytes      # delta_cnt = H(rho_cnt)
    merkle_path: List[Tuple[int, bytes]]
    open_root: bytes
    open_sig: bytes
    open_size: int
    open_index: int
    open_path: List[Tuple[int, bytes]]
    nonce: bytes
    slot_key: bytes            # rho_cnt  (this slot only)
    witness: bytes
    reason: str
    p: float
    coin: CoinTranscript

    def size_bytes(self) -> int:
        return (len(self.sid) + 1 + 8 + 4 + 4 + 8 + 4 + 4
                + len(self.batch_root) + len(self.batch_sig)
                + len(self.commitment) + len(self.key_commitment)
                + sum(1 + len(s) for _, s in self.merkle_path)
                + len(self.open_root) + len(self.open_sig)
                + sum(1 + len(s) for _, s in self.open_path)
                + len(self.nonce) + len(self.slot_key) + len(self.witness)
                + len(self.reason.encode())
                + self.coin.bytes_on_wire() + len(self.coin.batch_root))


@dataclass
class AuditReport:
    n_slots: int
    p: float
    marked: List[int]
    n_sacrificed: int
    caught: bool
    certificate: Optional[Certificate]
    yield_kept: float
    wire_bytes: int
    verify_seconds: float
    checks: Dict[str, int] = field(default_factory=dict)
    all_reasons: List[Tuple[int, str]] = field(default_factory=list)


def audit_batch(chip: Chip, counters: List[int], p: float,
                rng0: DeterministicRNG, rng1: DeterministicRNG,
                wellformed_spot_checks: int = 8,
                coin: Optional[CoinTranscript] = None,
                forced_marked: Optional[List[int]] = None,
                stop_on_first: bool = True) -> AuditReport:
    """Sacrificial cut-and-choose over a batch of READY slots (Algorithm 3).

    ``forced_marked`` exists only to model the counterfactual *oracle* chip in
    the adaptivity experiment; the honest protocol never uses it.
    """
    t_start = time.perf_counter()
    batch_root, batch_sig, tree = chip.sign_batch(counters)
    if coin is None:
        coin = coin_toss(batch_root, rng0, rng1)
    marked = (forced_marked if forced_marked is not None
              else challenge_set(coin.chi, counters, p))

    bundle = chip.open_batch(marked, coin.chi)

    wire = (coin.bytes_on_wire() + len(batch_root) + len(batch_sig)
            + bundle.wire_bytes)
    checks = {"opened": 0, "commit_fail": 0, "derivation_fail": 0,
              "wellformed_fail": 0, "keycommit_fail": 0}
    cert: Optional[Certificate] = None
    reasons: List[Tuple[int, str]] = []
    index_of = {c: i for i, c in enumerate(counters)}

    for cnt in bundle.order:
        op = bundle.openings[cnt]
        slot = chip.slots[cnt]
        checks["opened"] += 1
        reason = check_opened_slot(
            sid=chip.ctx.sid, cnt=cnt, witness=op.witness, nonce=op.nonce,
            rho_cnt=op.slot_key, gamma=slot.commitment,
            delta=slot.key_commitment, params=chip.ctx.params,
            spot_checks=wellformed_spot_checks, checks=checks)
        if reason is None:
            continue
        reasons.append((cnt, reason))
        if cert is None:
            idx = index_of[cnt]
            cert = Certificate(
                sid=chip.ctx.sid, party=chip.b, cnt=cnt, batch_index=idx,
                batch_root=batch_root, batch_sig=batch_sig,
                batch_counters_first=counters[0], batch_size=len(counters),
                commitment=slot.commitment, key_commitment=slot.key_commitment,
                merkle_path=tree.path(idx),
                open_root=bundle.root, open_sig=bundle.sig,
                open_size=len(bundle.order), open_index=bundle.index_of(cnt),
                open_path=bundle.tree.path(bundle.index_of(cnt)),
                nonce=op.nonce, slot_key=op.slot_key, witness=op.witness,
                reason=reason, p=p, coin=coin,
            )
            if stop_on_first:
                break

    elapsed = time.perf_counter() - t_start
    n_sac = checks["opened"]
    return AuditReport(
        n_slots=len(counters), p=p, marked=marked, n_sacrificed=n_sac,
        caught=cert is not None, certificate=cert,
        yield_kept=(len(counters) - n_sac) / len(counters),
        wire_bytes=wire, verify_seconds=elapsed, checks=checks,
        all_reasons=reasons,
    )


def check_opened_slot(sid: bytes, cnt: int, witness: bytes, nonce: bytes,
                      rho_cnt: bytes, gamma: bytes, delta: bytes,
                      params: DPFParams, spot_checks: int,
                      checks: Optional[Dict[str, int]] = None) -> Optional[str]:
    """The audit checks on one opened slot.  Returns a reason or ``None``."""
    checks = checks if checks is not None else {}
    def bump(k):
        checks[k] = checks.get(k, 0) + 1
    # (0) the revealed subkey must open the published key-commitment
    if slot_key_commit(rho_cnt) != delta:
        bump("keycommit_fail")
        return "slotkey-mismatch"
    # (i-a) the opened witness must open the published slot commitment
    digest = h(b"PACT|slotdigest", struct.pack(">Q", cnt), witness)
    if slot_commit(sid, cnt, digest, nonce) != gamma:
        bump("commit_fail")
        return "commitment-mismatch"
    # (i-b) ... and it must equal the honest derivation from rho_cnt
    honest = derive_template_from_slotkey(rho_cnt, cnt, params)
    if honest.witness_bytes() != witness:
        bump("derivation_fail")
        return "derivation-mismatch"
    # (ii) ... and it must be a well-formed payload-0 DPF for its declared target
    opened = parse_witness(witness, params)
    if opened is None:
        bump("wellformed_fail")
        return "malformed-template"
    if not template_is_wellformed(
            opened, spot_checks=spot_checks,
            rng=DeterministicRNG(b"wf" + struct.pack(">Q", cnt))):
        bump("wellformed_fail")
        return "malformed-template"
    return None


# ---------------------------------------------------------------------------
# Judgement: the third-party verifier.  Uses ONLY public data.
# ---------------------------------------------------------------------------


def judgement(cert: Certificate, attest_pub: bytes, params: DPFParams,
              spot_checks: int = 8) -> Tuple[bool, str]:
    """Verify a cheating certificate from public data alone.

    A third party holding nothing but the accused chip's attested public key
    checks, in order:
      (a) the batch root carries the accused chip's signature;
      (b) the accused slot's leaf (gamma_cnt, delta_cnt) is in that batch;
      (c) the coin transcript is well formed, bound to this batch root, and
          really marks this slot at the declared audit rate p;
      (d) the audit-round opening root carries the accused chip's signature
          and the presented opening is in it -- so the opening is the chip's
          own words, not a host's forgery;
      (e) rho_cnt opens delta_cnt and the witness opens gamma_cnt;
      (f) the witness differs from Derive(rho_cnt, cnt), or is not a
          well-formed payload-0 DPF.

    The chip is convicted only if (a)-(e) pass and (f) fires.  Failure of any
    of (a)-(e) rejects the certificate, which is what stops a malicious host
    from framing an honest chip.

    Returns ``(guilty, explanation)``.
    """
    # (a) attribution of the commitments
    msg = batch_message(cert.sid, [cert.batch_counters_first] * cert.batch_size,
                        cert.batch_root)
    if not attest_verify(attest_pub, msg, cert.batch_sig):
        return False, "batch signature invalid: certificate is not attributable"

    # (b) membership of the accused slot
    leaf = batch_leaf(cert.sid, cert.cnt, cert.commitment, cert.key_commitment)
    if not merkle_verify(leaf, cert.batch_index, cert.merkle_path, cert.batch_root):
        return False, "accused slot is not in the signed batch"

    # (c) the public coin really selected this slot
    if not cert.coin.verify():
        return False, "coin transcript malformed"
    if cert.coin.batch_root != cert.batch_root:
        return False, "coin was not bound to this batch root"
    if not slot_is_marked(cert.coin.chi, cert.batch_index, cert.cnt, cert.p):
        return False, "slot was not selected by the public coin"

    # (d) attribution of the opening
    omsg = opening_message(cert.sid, cert.coin.chi, cert.open_size, cert.open_root)
    if not attest_verify(attest_pub, omsg, cert.open_sig):
        return False, "opening signature invalid: the opening is not the chip's"
    oleaf = opening_leaf(cert.sid, cert.cnt, cert.slot_key, cert.nonce, cert.witness)
    if not merkle_verify(oleaf, cert.open_index, cert.open_path, cert.open_root):
        return False, "presented opening is not the one the chip signed"

    # (e) the opening opens what was committed
    if slot_key_commit(cert.slot_key) != cert.key_commitment:
        return True, "revealed slot subkey does not open the committed delta_cnt"
    digest = h(b"PACT|slotdigest", struct.pack(">Q", cert.cnt), cert.witness)
    if slot_commit(cert.sid, cert.cnt, digest, cert.nonce) != cert.commitment:
        return True, "opened witness does not open the committed slot gamma_cnt"

    # (f) recompute
    opened = parse_witness(cert.witness, params)
    if opened is None:
        return True, "witness does not parse as a template (structurally invalid)"
    honest = derive_template_from_slotkey(cert.slot_key, cert.cnt, params)
    if honest.witness_bytes() != cert.witness:
        return True, "opened template differs from Derive(rho_cnt, cnt)"
    if not template_is_wellformed(
            opened, spot_checks=spot_checks,
            rng=DeterministicRNG(b"wf" + struct.pack(">Q", cert.cnt))):
        return True, "opened template is not a well-formed payload-0 DPF"

    return False, "all checks pass: this certificate proves no cheating"


def parse_witness(w: bytes, params: DPFParams) -> Optional[Template]:
    from .dpf import CorrectionWord

    pre = b"PACT|tmpl|"
    if not w.startswith(pre):
        return None
    off = len(pre)
    nb = (params.n + 7) // 8
    mb = (params.m + 7) // 8
    try:
        cnt = int.from_bytes(w[off:off + 8], "big"); off += 8
        r = int.from_bytes(w[off:off + nb], "big"); off += nb
        s0 = w[off:off + LAMBDA_BYTES]; off += LAMBDA_BYTES
        s1 = w[off:off + LAMBDA_BYTES]; off += LAMBDA_BYTES
        cws = []
        for _ in range(params.n):
            s_cw = w[off:off + LAMBDA_BYTES]; off += LAMBDA_BYTES
            tl, tr = w[off], w[off + 1]; off += 2
            cws.append(CorrectionWord(s_cw, tl, tr))
        fcw0 = int.from_bytes(w[off:off + mb], "big"); off += mb
        sigma = 1 if w[off] == 0 else -1; off += 1
        if off != len(w):
            return None
        return Template(cnt=cnt, r=r, s0_0=s0, s0_1=s1, cws=tuple(cws),
                        fcw0=fcw0, sigma=sigma, params=params)
    except (IndexError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Forged-accusation attempts (the empirical side of defamation freeness)
# ---------------------------------------------------------------------------

FORGERY_STRATEGIES = (
    "tamper-witness",
    "tamper-slotkey",
    "tamper-nonce",
    "swap-opening",
    "replay-coin",
    "unmarked-slot",
    "forge-batch-signature",
    "forge-opening-signature",
    "wrong-slot-index",
)


def forge_certificate(chip: Chip, counters: List[int], p: float,
                      rng0: DeterministicRNG, rng1: DeterministicRNG,
                      strategy: str) -> Certificate:
    """A malicious *host* tries to frame an honest chip.

    The host sees every public value and relays every message, so it can try
    to alter, replay, re-index or re-sign any part of the transcript.  Each
    strategy below is one such attempt; ``judgement`` must reject all of them.
    """
    from .primitives import AttestationKey

    batch_root, batch_sig, tree = chip.sign_batch(counters)
    coin = coin_toss(batch_root, rng0, rng1)
    marked = challenge_set(coin.chi, counters, p)
    if not marked:
        marked = [counters[0]]
    bundle = chip.open_batch(marked, coin.chi)

    cnt = bundle.order[0]
    idx = counters.index(cnt)
    slot = chip.slots[cnt]
    op = bundle.openings[cnt]
    witness, rho_cnt, nonce = op.witness, op.slot_key, op.nonce
    open_index, open_path = bundle.index_of(cnt), bundle.tree.path(bundle.index_of(cnt))
    open_root, open_sig = bundle.root, bundle.sig

    if strategy == "tamper-witness":
        witness = witness[:-1] + bytes([witness[-1] ^ 1])
    elif strategy == "tamper-slotkey":
        rho_cnt = bytes([rho_cnt[0] ^ 1]) + rho_cnt[1:]
    elif strategy == "tamper-nonce":
        nonce = bytes([nonce[0] ^ 1]) + nonce[1:]
    elif strategy == "swap-opening":
        if len(bundle.order) > 1:
            other = bundle.openings[bundle.order[1]]
            witness, rho_cnt, nonce = other.witness, other.slot_key, other.nonce
        else:
            witness = witness[:-1] + bytes([witness[-1] ^ 2])
    elif strategy == "replay-coin":
        coin = coin_toss(h(b"other-batch"), rng0, rng1)
    elif strategy == "unmarked-slot":
        unmarked = [c for i, c in enumerate(counters)
                    if not slot_is_marked(coin.chi, i, c, p)]
        cnt = unmarked[0]
        idx = counters.index(cnt)
        slot = chip.slots[cnt]
        witness = slot.template.witness_bytes()
        rho_cnt, nonce = slot.slot_key, slot.nonce
    elif strategy == "forge-batch-signature":
        rogue = AttestationKey.generate(DeterministicRNG(b"rogue"))
        batch_sig = rogue.sign(batch_message(chip.ctx.sid, counters, batch_root))
    elif strategy == "forge-opening-signature":
        rogue = AttestationKey.generate(DeterministicRNG(b"rogue2"))
        open_sig = rogue.sign(opening_message(chip.ctx.sid, coin.chi,
                                              len(bundle.order), open_root))
    elif strategy == "wrong-slot-index":
        idx = (idx + 1) % len(counters)
    else:
        raise ValueError(strategy)

    return Certificate(
        sid=chip.ctx.sid, party=chip.b, cnt=cnt, batch_index=idx,
        batch_root=batch_root, batch_sig=batch_sig,
        batch_counters_first=counters[0], batch_size=len(counters),
        commitment=slot.commitment, key_commitment=slot.key_commitment,
        merkle_path=tree.path(idx),
        open_root=open_root, open_sig=open_sig, open_size=len(bundle.order),
        open_index=open_index, open_path=open_path,
        nonce=nonce, slot_key=rho_cnt, witness=witness,
        reason="forged", p=p, coin=coin,
    )


# ---------------------------------------------------------------------------
# Public cross-chip consistency (deterministic single-chip detection).
# ---------------------------------------------------------------------------
# Both chips share the pairing root and derive per-slot material
# deterministically, so for every slot counter their published commitments
# (gamma_cnt, delta_cnt) must be *identical*.  A single deviating chip --
# vendor target, vendor subkey, biased seed, anything -- breaks that equality
# and is caught deterministically, at zero sacrifice, from public data alone.
# The sacrificial cut-and-choose of Algorithm 3 remains the deterrent for the
# case the consistency check cannot see: chips that collude (one vendor on
# both boards) publish identical commitments and are caught only at rate p.


@dataclass
class ConsistencyCertificate:
    """A transferable conflict certificate: two signed batch leaves that
    disagree on the same (sid, cnt)."""
    sid: bytes
    cnt: int
    batch_index0: int
    batch_index1: int
    batch_size0: int
    batch_size1: int
    counters_first0: int
    counters_first1: int
    root0: bytes
    sig0: bytes
    root1: bytes
    sig1: bytes
    path0: List[Tuple[int, bytes]]
    path1: List[Tuple[int, bytes]]
    commitment0: bytes
    key_commitment0: bytes
    commitment1: bytes
    key_commitment1: bytes
    reason: str

    def size_bytes(self) -> int:
        return (len(self.sid) + 8 + 4 + 4 + 4 + 4 + 8 + 8
                + len(self.root0) + len(self.sig0)
                + len(self.root1) + len(self.sig1)
                + sum(1 + len(s) for _, s in self.path0)
                + sum(1 + len(s) for _, s in self.path1)
                + len(self.commitment0) + len(self.key_commitment0)
                + len(self.commitment1) + len(self.key_commitment1)
                + len(self.reason.encode()))


def cross_chip_consistency(chip0: "Chip", chip1: "Chip", counters: List[int]
                           ) -> List[Tuple[int, str]]:
    """Compare the public per-slot commitments of the two chips.  Returns
    [(cnt, reason), ...] for every disagreeing slot; empty when they agree."""
    out: List[Tuple[int, str]] = []
    for cnt in counters:
        s0, s1 = chip0.slots[cnt], chip1.slots[cnt]
        if s0.key_commitment != s1.key_commitment:
            out.append((cnt, "subkey-commitment-mismatch"))
        elif s0.commitment != s1.commitment:
            out.append((cnt, "template-commitment-mismatch"))
    return out


def cross_chip_consistency_audit(chip0: "Chip", chip1: "Chip",
                                 counters: List[int]) -> Tuple[bool, Optional[ConsistencyCertificate], List[Tuple[int, str]]]:
    """Sign both batches and build a transferable conflict certificate for the
    first disagreeing slot.  Returns (caught, certificate, all_violations)."""
    root0, sig0, tree0 = chip0.sign_batch(counters)
    root1, sig1, tree1 = chip1.sign_batch(counters)
    violations = cross_chip_consistency(chip0, chip1, counters)
    cert = None
    if violations:
        cnt, reason = violations[0]
        i0 = counters.index(cnt)
        i1 = counters.index(cnt)
        s0, s1 = chip0.slots[cnt], chip1.slots[cnt]
        cert = ConsistencyCertificate(
            sid=chip0.ctx.sid, cnt=cnt,
            batch_index0=i0, batch_index1=i1,
            batch_size0=len(counters), batch_size1=len(counters),
            counters_first0=counters[0], counters_first1=counters[0],
            root0=root0, sig0=sig0, root1=root1, sig1=sig1,
            path0=tree0.path(i0), path1=tree1.path(i1),
            commitment0=s0.commitment, key_commitment0=s0.key_commitment,
            commitment1=s1.commitment, key_commitment1=s1.key_commitment,
            reason=reason,
        )
    return cert is not None, cert, violations


def consistency_judgement(cert: ConsistencyCertificate, attest_pub0: bytes,
                          attest_pub1: bytes) -> Tuple[bool, str]:
    """Third-party verification of a conflict certificate from public data.

    Both signed batch roots must verify, both leaves must be members of their
    batches at the *same* slot counter, and the committed values must differ.
    The certificate convicts the *pair*: two chips bound to the same pairing
    root cannot honestly disagree, so at least one deviated.
    """
    msg0 = batch_message(cert.sid, [cert.counters_first0] * cert.batch_size0, cert.root0)
    if not attest_verify(attest_pub0, msg0, cert.sig0):
        return False, "chip-0 batch signature invalid"
    msg1 = batch_message(cert.sid, [cert.counters_first1] * cert.batch_size1, cert.root1)
    if not attest_verify(attest_pub1, msg1, cert.sig1):
        return False, "chip-1 batch signature invalid"
    leaf0 = batch_leaf(cert.sid, cert.cnt, cert.commitment0, cert.key_commitment0)
    leaf1 = batch_leaf(cert.sid, cert.cnt, cert.commitment1, cert.key_commitment1)
    if not merkle_verify(leaf0, cert.batch_index0, cert.path0, cert.root0):
        return False, "chip-0 leaf is not in its signed batch"
    if not merkle_verify(leaf1, cert.batch_index1, cert.path1, cert.root1):
        return False, "chip-1 leaf is not in its signed batch"
    if cert.commitment0 == cert.commitment1 and cert.key_commitment0 == cert.key_commitment1:
        return False, "chips agree: this certificate proves no conflict"
    return True, "the two chips committed different material for the same slot"
