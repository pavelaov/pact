"""
PACT reference implementation -- concrete adversaries.

A threat model is only as convincing as the adversaries one is willing to
write down and run.  This module implements them as *executable* attackers,
not as Bernoulli abstractions, so that every detection number in the paper is
produced by running the real GEN/BIND/EVAL/audit code against a real attack.

Adversaries implemented
-----------------------

``VendorBackdoorChip``
    The strongest covert generator we know how to build against a
    template-based FSS key generator, and the subject of the paper's new
    experiment.  Instead of deriving the template target from the per-slot
    subkey, the chip derives it from a *vendor* key:

        r_cnt = PRF_{k_v}(sid, cnt)

    The template is then a perfectly well-formed payload-0 DPF, the bound key
    is perfectly correct, and every value the hosts ever see is pseudorandom --
    yet anybody who knows k_v recovers the secret index of every real query
    from the *public* shift alone:

        Delta = r - alpha   ==>   alpha = PRF_{k_v}(sid,cnt) - Delta.

    This defeats attestation, correctness testing, output inspection and any
    statistical test on the generator's outputs.  It is caught only by a check
    that recomputes the derivation against a commitment made *before the input
    existed* -- i.e. exactly by PACT's audit.

``BiasedSeedChip`` / ``CheatProfile`` modes
    Weaker classical backdoors (predictable root seed, corrupted correction
    word, corrupted final word), used for the cross-mode detection table.

``oracle_taint_choice`` / ``blind_taint_choice``
    The counterfactual adaptive chip that learns the audit challenge before
    committing, versus the real chip that must commit first.  Isolates the
    value of the commit-then-challenge ordering.

``MaliciousBindHost``
    A host that commits one input share at submit time and binds a different
    one (``protocol.Host(cheat_bind=True)``).
"""

from __future__ import annotations

import hashlib
import hmac
import struct
from dataclasses import dataclass
from typing import List, Optional, Sequence, Set

from .dpf import DPFParams, Template, derive_template_from_slotkey, gen_template_from_randomness
from .primitives import DeterministicRNG, LAMBDA_BYTES, kdf
from .protocol import CheatProfile, Chip, PairingContext

CHEAT_MODES = ("vendor", "seed", "target", "cw", "fcw")


# ---------------------------------------------------------------------------
# The vendor-PRF backdoor
# ---------------------------------------------------------------------------


def vendor_target(vendor_key: bytes, sid: bytes, cnt: int, params: DPFParams) -> int:
    """The backdoored target: predictable to the vendor, pseudorandom to all."""
    d = hmac.new(vendor_key, b"PACT|vendor|" + sid + struct.pack(">Q", cnt),
                 hashlib.sha256).digest()
    return int.from_bytes(d[:8], "big") % params.domain


class VendorBackdoorChip(Chip):
    """A chip that plants a vendor-predictable target in a chosen slot subset."""

    def __init__(self, b: int, ctx: PairingContext, attest, vendor_key: bytes,
                 taint_slots: Sequence[int], rng: Optional[DeterministicRNG] = None):
        super().__init__(b, ctx, attest,
                         cheat=CheatProfile(frozenset(taint_slots), "vendor"), rng=rng)
        self.vendor_key = vendor_key
        self.taint: Set[int] = set(taint_slots)

    def _template_for(self, cnt: int, rho_cnt: bytes) -> Template:
        if cnt not in self.taint:
            return super()._template_for(cnt, rho_cnt)
        # Same seeds as the honest derivation, so the template is a valid
        # payload-0 DPF and nothing about it looks wrong.  Only the *target*
        # is replaced by a value the vendor can predict.
        p = self.ctx.params
        nb = (p.n + 7) // 8
        material = kdf(rho_cnt, b"PACT|template", cnt, nb + 2 * LAMBDA_BYTES)
        s0_0 = material[nb:nb + LAMBDA_BYTES]
        s0_1 = material[nb + LAMBDA_BYTES:nb + 2 * LAMBDA_BYTES]
        r = vendor_target(self.vendor_key, self.ctx.sid, cnt, p)
        return gen_template_from_randomness(cnt, r, s0_0, s0_1, p)


def vendor_recover_alpha(vendor_key: bytes, sid: bytes, cnt: int, delta: int,
                         params: DPFParams) -> int:
    """What the vendor computes from PUBLIC data only.

    ``delta`` is the public domain shift that BIND emits; the vendor needs no
    key share, no host state and no network position beyond seeing ``delta``.
    """
    return (vendor_target(vendor_key, sid, cnt, params) - delta) % params.domain


# ---------------------------------------------------------------------------
# The vendor-SUBKEY backdoor: subkey-level deviation on a single chip.
# ---------------------------------------------------------------------------
# Instead of replacing only the template target, the chip replaces the per-slot
# SUBKEY itself: rho'_cnt = PRF_{k_v}(sid, cnt), and derives the whole template
# honestly from rho'_cnt.  Every internal audit check -- H(rho'_cnt)=delta'_cnt,
# Com(...)=gamma'_cnt, Derive(rho'_cnt,cnt)=T'_cnt, well-formedness -- passes,
# because the deviation sits one level *above* everything the sacrificial audit
# recomputes.  It is caught only by a check that pins the subkey derivation to
# the shared pairing root: the public cross-chip consistency of delta_cnt (and
# gamma_cnt), which a *single* deviating chip fails deterministically.  With
# both chips backdoored (one vendor), consistency is silent and the sacrificial
# audit remains the deterrent.
# ---------------------------------------------------------------------------


class VendorSubkeyBackdoorChip(Chip):
    """A chip whose per-slot subkey is vendor-predictable; internally consistent."""

    def __init__(self, b: int, ctx: PairingContext, attest, vendor_key: bytes,
                 taint_slots: Sequence[int], rng: Optional[DeterministicRNG] = None):
        super().__init__(b, ctx, attest,
                         cheat=CheatProfile(frozenset(taint_slots), "vendor"), rng=rng)
        self.vendor_key = vendor_key
        self.taint: Set[int] = set(taint_slots)

    def _subkey_for(self, cnt: int) -> bytes:
        if cnt not in self.taint:
            return super()._subkey_for(cnt)
        return kdf(self.vendor_key, b"PACT|vendorslot", cnt, 32)

    def _template_for(self, cnt: int, rho_cnt: bytes) -> Template:
        # The deviation lives entirely in the subkey: the template is derived
        # *honestly* from the vendor-chosen subkey, so every internal audit
        # check stays consistent (that is the point of this adversary).
        return derive_template_from_slotkey(rho_cnt, cnt, self.ctx.params)


# ---------------------------------------------------------------------------
# Adaptive (oracle) chip -- counterfactual, isolates the ordering's value
# ---------------------------------------------------------------------------


@dataclass
class AdaptivePlan:
    """Where a chip places its taints."""
    taints: List[int]


def oracle_taint_choice(counters: Sequence[int], marked: Sequence[int], c: int
                        ) -> AdaptivePlan:
    """An oracle chip places every taint on a slot it knows will NOT be opened."""
    marked_set = set(marked)
    safe = [x for x in counters if x not in marked_set]
    return AdaptivePlan(taints=list(safe[:c]))


def blind_taint_choice(counters: Sequence[int], c: int, rng: DeterministicRNG
                       ) -> AdaptivePlan:
    """A real chip must choose blind: it commits before the coin is tossed."""
    pool = list(counters)
    picks: List[int] = []
    for _ in range(min(c, len(pool))):
        picks.append(pool.pop(rng.randint(len(pool))))
    return AdaptivePlan(taints=picks)
