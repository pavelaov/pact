"""
PACT reference implementation -- distributed point functions and *templates*.

This module implements the tree-based DPF of Boyle, Gilboa and Ishai (CCS'16)
and, on top of it, the object PACT's whole argument rests on: the
**commitment template**, a payload-zero DPF for a chip-chosen pseudorandom
target that is derived, committed and stored *before any input exists*.

The structural fact the paper uses is made explicit here in code:

    FCW(beta) = sigma * beta + FCW0    (mod 2^m),      sigma in {+1,-1}

Every word of a DPF key except the final correction word is a function of the
seeds and the target only; the payload enters exactly once, affinely.  A
template stores (seeds, {CW}, FCW0, sigma, r) -- all input-independent -- and
BIND turns it into a real key with the single affine patch above plus a public
domain shift.

Group: Z_{2^m}.  Domain: {0,1}^n.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import List, Optional, Tuple

from .primitives import LAMBDA_BYTES, DeterministicRNG, h, kdf, prg


def _bit(x: int, i: int, n: int) -> int:
    """i-th bit of x from the most significant end (i = 1 .. n)."""
    return (x >> (n - i)) & 1


def _xor(a: bytes, b: bytes) -> bytes:
    return bytes(u ^ v for u, v in zip(a, b))


def convert(seed: bytes, m_bits: int) -> int:
    """Convert: {0,1}^lambda -> Z_{2^m}."""
    return int.from_bytes(seed[: (m_bits + 7) // 8], "big") & ((1 << m_bits) - 1)


@dataclass(frozen=True)
class CorrectionWord:
    s_cw: bytes
    t_cw_l: int
    t_cw_r: int

    def to_bytes(self) -> bytes:
        return self.s_cw + bytes([self.t_cw_l, self.t_cw_r])


@dataclass(frozen=True)
class DPFParams:
    n: int = 16       # domain bits  ({0,1}^n)
    m: int = 32       # group bits   (Z_{2^m})

    @property
    def domain(self) -> int:
        return 1 << self.n

    @property
    def modulus(self) -> int:
        return 1 << self.m


@dataclass(frozen=True)
class DPFKey:
    """One party's DPF key share."""
    b: int
    s0: bytes
    cws: Tuple[CorrectionWord, ...]
    fcw: int
    params: DPFParams

    def to_bytes(self) -> bytes:
        return (
            bytes([self.b])
            + self.s0
            + b"".join(cw.to_bytes() for cw in self.cws)
            + self.fcw.to_bytes((self.params.m + 7) // 8, "big")
        )

    @property
    def size_bytes(self) -> int:
        return len(self.to_bytes())


@dataclass(frozen=True)
class Template:
    """A committed, input-independent DPF template (the object PACT audits).

    ``r``       pseudorandom target chosen by the chip from (rho, cnt)
    ``s0_0/s0_1`` the two root seeds (both derived from rho, so *both* chips
                know both; each chip keeps only its own share of the key)
    ``cws``     per-level correction words -- payload independent
    ``fcw0``    the payload-independent part of the final correction word
    ``sigma``   +1 / -1, the affine coefficient of the payload
    """
    cnt: int
    r: int
    s0_0: bytes
    s0_1: bytes
    cws: Tuple[CorrectionWord, ...]
    fcw0: int
    sigma: int
    params: DPFParams

    # ---- serialisation: this is *exactly* what an auditor recomputes -------
    def witness_bytes(self) -> bytes:
        mb = (self.params.m + 7) // 8
        return (
            b"PACT|tmpl|"
            + self.cnt.to_bytes(8, "big")
            + self.r.to_bytes((self.params.n + 7) // 8, "big")
            + self.s0_0
            + self.s0_1
            + b"".join(cw.to_bytes() for cw in self.cws)
            + (self.fcw0 % self.params.modulus).to_bytes(mb, "big")
            + bytes([0 if self.sigma == 1 else 1])
        )

    @property
    def witness_size_bytes(self) -> int:
        return len(self.witness_bytes())

    def key_share(self, b: int, beta: int = 0) -> DPFKey:
        """The party-b key of this template, with payload ``beta`` patched in."""
        fcw = (self.sigma * beta + self.fcw0) % self.params.modulus
        return DPFKey(
            b=b,
            s0=self.s0_0 if b == 0 else self.s0_1,
            cws=self.cws,
            fcw=fcw,
            params=self.params,
        )


# ---------------------------------------------------------------------------
# Core DPF generation, factored so that the payload enters exactly once
# ---------------------------------------------------------------------------


def gen_template_from_randomness(
    cnt: int, r: int, s0_0: bytes, s0_1: bytes, params: DPFParams
) -> Template:
    """Deterministic template generation (this *is* Derive(rho, cnt)).

    Runs the BGI16 tree construction for target ``r`` with payload 0, and
    keeps the affine decomposition of the final correction word.
    """
    n, m = params.n, params.m
    mod = params.modulus

    s0, s1 = s0_0, s0_1
    t0, t1 = 0, 1
    cws: List[CorrectionWord] = []

    for i in range(1, n + 1):
        sL0, tL0, sR0, tR0 = prg(s0)
        sL1, tL1, sR1, tR1 = prg(s1)
        a = _bit(r, i, n)
        if a == 0:
            keep0, keep1 = sL0, sL1
            tkeep0, tkeep1 = tL0, tL1
            lose0, lose1 = sR0, sR1
        else:
            keep0, keep1 = sR0, sR1
            tkeep0, tkeep1 = tR0, tR1
            lose0, lose1 = sL0, sL1

        s_cw = _xor(lose0, lose1)
        t_cw_l = tL0 ^ tL1 ^ a ^ 1
        t_cw_r = tR0 ^ tR1 ^ a
        cws.append(CorrectionWord(s_cw, t_cw_l, t_cw_r))

        t_cw_keep = t_cw_l if a == 0 else t_cw_r
        s0 = _xor(keep0, s_cw) if t0 else keep0
        s1 = _xor(keep1, s_cw) if t1 else keep1
        t0 = tkeep0 ^ (t0 & t_cw_keep)
        t1 = tkeep1 ^ (t1 & t_cw_keep)

    sigma = -1 if t1 == 1 else 1
    fcw0 = (sigma * (-convert(s0, m) + convert(s1, m))) % mod

    return Template(
        cnt=cnt, r=r, s0_0=s0_0, s0_1=s0_1,
        cws=tuple(cws), fcw0=fcw0, sigma=sigma, params=params,
    )


def derive_template_from_slotkey(rho_cnt: bytes, cnt: int, params: DPFParams) -> Template:
    """``Derive(rho_cnt, cnt)``: the honest, deterministic template derivation.

    Both paired chips run exactly this on the same per-slot subkey and
    therefore obtain identical templates with **zero communication** -- the
    defining property of GEN.  Because the derivation takes only ``rho_cnt``,
    a third party who is shown ``rho_cnt`` when a slot is sacrificed can redo
    it and convict a cheating chip without ever learning ``rho``.
    """
    nb = (params.n + 7) // 8
    material = kdf(rho_cnt, b"PACT|template", cnt, nb + 2 * LAMBDA_BYTES)
    r = int.from_bytes(material[:nb], "big") % params.domain
    s0_0 = material[nb : nb + LAMBDA_BYTES]
    s0_1 = material[nb + LAMBDA_BYTES : nb + 2 * LAMBDA_BYTES]
    return gen_template_from_randomness(cnt, r, s0_0, s0_1, params)


def derive_template(root: bytes, cnt: int, params: DPFParams) -> Template:
    """Convenience wrapper: full two-level derivation from the pairing root."""
    from .primitives import slot_key
    return derive_template_from_slotkey(slot_key(root, cnt), cnt, params)


def dpf_gen(alpha: int, beta: int, params: DPFParams, rng: DeterministicRNG
            ) -> Tuple[DPFKey, DPFKey]:
    """Plain (non-PACT) DPF key generation -- used by tests and baselines."""
    s0_0 = rng.bytes(LAMBDA_BYTES)
    s0_1 = rng.bytes(LAMBDA_BYTES)
    tmpl = gen_template_from_randomness(0, alpha, s0_0, s0_1, params)
    return tmpl.key_share(0, beta), tmpl.key_share(1, beta)


def dpf_eval(key: DPFKey, x: int) -> int:
    """Evaluate one DPF key share at ``x``."""
    p = key.params
    n, m, mod = p.n, p.m, p.modulus
    s, t = key.s0, key.b
    for i in range(1, n + 1):
        sL, tL, sR, tR = prg(s)
        cw = key.cws[i - 1]
        if _bit(x, i, n) == 0:
            s = _xor(sL, cw.s_cw) if t else sL
            t = tL ^ (t & cw.t_cw_l)
        else:
            s = _xor(sR, cw.s_cw) if t else sR
            t = tR ^ (t & cw.t_cw_r)
    sign = -1 if key.b == 1 else 1
    return (sign * (convert(s, m) + t * key.fcw)) % mod


def dpf_eval_full(key: DPFKey) -> List[int]:
    """Full-domain evaluation in O(2^n) PRG calls instead of O(n*2^n).

    Expands the whole GGM tree breadth-first, reusing each internal node once.
    Used by the correctness experiments, which check every domain point of
    every generated key rather than sampling.
    """
    p = key.params
    n, m, mod = p.n, p.m, p.modulus
    level = [(key.s0, key.b)]
    for i in range(1, n + 1):
        cw = key.cws[i - 1]
        nxt = []
        for s, t in level:
            sL, tL, sR, tR = prg(s)
            if t:
                nxt.append((_xor(sL, cw.s_cw), tL ^ cw.t_cw_l))
                nxt.append((_xor(sR, cw.s_cw), tR ^ cw.t_cw_r))
            else:
                nxt.append((sL, tL))
                nxt.append((sR, tR))
        level = nxt
    sign = -1 if key.b == 1 else 1
    return [(sign * (convert(s, m) + t * key.fcw)) % mod for s, t in level]


# ---------------------------------------------------------------------------
# Template well-formedness check (the second half of the audit)
# ---------------------------------------------------------------------------


def template_is_wellformed(tmpl: Template, spot_checks: int = 8,
                           rng: Optional[DeterministicRNG] = None,
                           exhaustive: bool = False) -> bool:
    """Check that a template really is a payload-0 DPF for its declared target.

    This is check (ii) of the audit.  It re-expands the tree from the opened
    witness and verifies the DPF invariant: the two shares cancel everywhere
    off the declared target and reconstruct the payload (0) on it.

    Two regimes, both reported in the paper:
      * ``exhaustive=True``  -- full-domain check.  Sound, and what the paper's
        small-domain correctness experiment uses.
      * default              -- the target plus ``spot_checks`` random
        off-target points.  This is what an auditor with a bounded time budget
        runs at deployment scale; a malformed template that survives it must
        agree with a valid DPF on every probed point.
    """
    p = tmpl.params
    k0 = tmpl.key_share(0, 0)
    k1 = tmpl.key_share(1, 0)
    mod = p.modulus

    if (dpf_eval(k0, tmpl.r) + dpf_eval(k1, tmpl.r)) % mod != 0:
        return False

    if exhaustive:
        for x in range(p.domain):
            if x == tmpl.r:
                continue
            if (dpf_eval(k0, x) + dpf_eval(k1, x)) % mod != 0:
                return False
        return True

    rng = rng or DeterministicRNG(b"wellformed")
    for _ in range(spot_checks):
        x = rng.randint(p.domain)
        if x == tmpl.r:
            continue
        if (dpf_eval(k0, x) + dpf_eval(k1, x)) % mod != 0:
            return False
    return True


def recompute_and_compare(rho_cnt: bytes, cnt: int, opened: Template) -> bool:
    """Check (i) of the audit: does the opened witness equal Derive(rho_cnt,cnt)?"""
    honest = derive_template_from_slotkey(rho_cnt, cnt, opened.params)
    return honest.witness_bytes() == opened.witness_bytes()


def template_fingerprint(tmpl: Template) -> bytes:
    return h(b"PACT|tmplfp", tmpl.witness_bytes())


def tamper_template(tmpl: Template, rng: DeterministicRNG, mode: str = "seed") -> Template:
    """Produce a *deviating* template (what a covert chip would emit).

    ``mode``:
      ``seed``   -- a biased root seed (the classic backdoor: a predictable seed)
      ``target`` -- a biased target r (steers which point the key hides)
      ``cw``     -- a corrupted correction word (breaks the DPF invariant)
      ``fcw``    -- a corrupted payload-independent final word
    """
    p = tmpl.params
    if mode == "seed":
        bad = rng.bytes(LAMBDA_BYTES)
        return gen_template_from_randomness(tmpl.cnt, tmpl.r, bad, tmpl.s0_1, p)
    if mode == "target":
        r2 = (tmpl.r + 1 + rng.randint(max(1, p.domain - 1))) % p.domain
        return gen_template_from_randomness(tmpl.cnt, r2, tmpl.s0_0, tmpl.s0_1, p)
    if mode == "cw":
        idx = rng.randint(len(tmpl.cws))
        cw = tmpl.cws[idx]
        flipped = CorrectionWord(_xor(cw.s_cw, b"\x01" + b"\x00" * (LAMBDA_BYTES - 1)),
                                 cw.t_cw_l, cw.t_cw_r)
        cws = list(tmpl.cws)
        cws[idx] = flipped
        return replace(tmpl, cws=tuple(cws))
    if mode == "fcw":
        return replace(tmpl, fcw0=(tmpl.fcw0 + 1) % p.modulus)
    raise ValueError(f"unknown tamper mode {mode!r}")
