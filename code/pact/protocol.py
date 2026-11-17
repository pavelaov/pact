"""
PACT reference implementation -- the GEN / BIND / EVAL protocol.

Deployment picture (this is the whole point of the paper: *where* the work
happens decides whether it can be audited):

    party P_b  =  untrusted host H_b   +   fixed-function chip C_b
    the two chips never talk directly; every chip-to-chip message is relayed
    host-to-host over an authenticated channel keyed at pairing time.

    PAIR  (once, input-independent)   -> shared root rho, channel key K_ch,
                                        monotone counter, attested key pair
    GEN   (offline, zero messages)    -> committed templates
    BIND  (exactly one exchange)      -> a real key, via ONE affine patch
    EVAL  (local, zero messages)      -> FSS evaluation

Interfaces are typed and every message is a real byte string, so the
communication numbers in the paper are measured, not estimated.

Per-slot derivation is two-level, from a *ratcheted* root (forward erasure):

    rho_cnt = KDF(root_live, "slotkey", cnt)        (per-slot subkey)
    template, nonce, bind masks  <- rho_cnt
    root_live <- one-way ratchet after every BIND  (past slots unrecoverable)

with the public commitments

    gamma_cnt = Com(sid, cnt, nonce_cnt, digest(template))
    delta_cnt = H(rho_cnt)

both published at GEN and both covered by the batch attestation signature.
Opening a sacrificed slot reveals rho_cnt, which lets *anyone* recompute the
template and check it, and reveals nothing about the root or any other slot.

BIND transport: nonce || AES-CTR(body, K_enc) || CMAC(all, K_mac),
with K_enc and K_mac derived from the pairing channel key.  A malicious relay
host reads nothing; the two masked shares can no longer be summed to recover
the input.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from .dpf import (
    DPFKey,
    DPFParams,
    Template,
    derive_template_from_slotkey,
    dpf_eval,
    tamper_template,
)
from .primitives import (
    AttestationKey,
    DeterministicRNG,
    LAMBDA_BYTES,
    MerkleTree,
    channel_decrypt,
    channel_encrypt,
    channel_mac,
    channel_mac_verify,
    derive_channel_keys,
    h,
    kdf,
    ratchet,
    slot_commit,
    slot_commit_verify,
    slot_key,
    slot_key_commit,
)


class SlotState(Enum):
    READY = "ready"
    SACRIFICED = "sacrificed"
    BINDING = "binding"
    USED = "used"


@dataclass
class Slot:
    cnt: int
    template: Template
    slot_key: bytes
    nonce: bytes
    commitment: bytes        # gamma_cnt   (public, hiding, binding)
    key_commitment: bytes    # delta_cnt = H(rho_cnt)
    state: SlotState = SlotState.READY

    def leaf(self, sid: bytes) -> bytes:
        """The Merkle leaf that the batch attestation signs."""
        return h(b"PACT|leaf", sid, struct.pack(">Q", self.cnt),
                 self.commitment, self.key_commitment)


def batch_leaf(sid: bytes, cnt: int, commitment: bytes, key_commitment: bytes) -> bytes:
    return h(b"PACT|leaf", sid, struct.pack(">Q", cnt), commitment, key_commitment)


@dataclass
class PairingContext:
    """Output of the one-time attested pairing (input-independent)."""
    root: bytes
    channel_key: bytes
    channel_enc_key: bytes          # confidential BIND channel
    channel_mac_key: bytes          # derived MAC key (enc-then-MAC)
    sid: bytes
    params: DPFParams
    attest_pub: Dict[int, bytes] = field(default_factory=dict)


def pair_chips(rng: DeterministicRNG, params: DPFParams
               ) -> Tuple[PairingContext, Dict[int, AttestationKey]]:
    root = rng.bytes(32)
    channel_key = rng.bytes(16)
    enc_key, mac_key = derive_channel_keys(channel_key)
    sid = rng.bytes(16)
    keys = {0: AttestationKey.generate(rng), 1: AttestationKey.generate(rng)}
    ctx = PairingContext(root=root, channel_key=channel_key,
                         channel_enc_key=enc_key, channel_mac_key=mac_key,
                         sid=sid, params=params,
                         attest_pub={b: keys[b].public_bytes for b in (0, 1)})
    return ctx, keys


# ---------------------------------------------------------------------------
# The chip
# ---------------------------------------------------------------------------


@dataclass
class CheatProfile:
    """How (and how often) a covert chip deviates during GEN."""
    taint_slots: frozenset = frozenset()
    mode: str = "seed"


class Chip:
    """A fixed-function PACT chip.

    The chip is the *trusted* component; the audit is what makes that trust
    checkable.  A ``CheatProfile`` turns the same object into the covert
    adversary of the threat model, so the honest and the cheating chip run the
    identical code path and the audit cannot distinguish them structurally.
    """

    def __init__(self, b: int, ctx: PairingContext, attest: AttestationKey,
                 cheat: Optional[CheatProfile] = None,
                 rng: Optional[DeterministicRNG] = None):
        self.b = b
        self.ctx = ctx
        self.attest = attest
        self.cheat = cheat or CheatProfile()
        self.rng = rng or DeterministicRNG(b"chip%d" % b)
        self.slots: Dict[int, Slot] = {}
        self._live_root = ctx.root
        self.stats = {"gen": 0, "bind": 0, "opened": 0, "aborts": 0,
                      "bind_bytes": 0, "open_bytes": 0, "sign": 0,
                      "exposure_aes": 0, "exposure_ops": 0}
        # transient bind region H_bind: the ONLY place a reconstructed input lives
        self._hbind: Optional[Tuple[int, int]] = None

    # -- GEN ---------------------------------------------------------------
    def _subkey_for(self, cnt: int) -> bytes:
        """Hook: the per-slot subkey (covert chips override for subkey-level
        backdoors that keep every internal check consistent)."""
        return slot_key(self._live_root, cnt)

    def _template_for(self, cnt: int, rho_cnt: bytes) -> Template:
        """Hook: subclasses (covert chips) override what GEN actually emits."""
        tmpl = derive_template_from_slotkey(rho_cnt, cnt, self.ctx.params)
        if cnt in self.cheat.taint_slots:
            tmpl = tamper_template(tmpl, self.rng, self.cheat.mode)
        return tmpl

    def gen(self, cnt: int) -> Slot:
        """Offline, zero-communication committed template generation.

        Forward erasure: the per-slot subkey derives from the *live
        ratcheted root*, not from the pairing root.  The root advances one-way
        at every BIND, so a chip compromised after a slot is consumed cannot
        rederive that slot's (or any earlier slot's) subkey, target or input.
        """
        rho_cnt = self._subkey_for(cnt)
        tmpl = self._template_for(cnt, rho_cnt)
        nonce = kdf(rho_cnt, b"PACT|slotnonce", cnt, 16)
        digest = h(b"PACT|slotdigest", struct.pack(">Q", cnt), tmpl.witness_bytes())
        gamma = slot_commit(self.ctx.sid, cnt, digest, nonce)
        delta = slot_key_commit(rho_cnt)
        slot = Slot(cnt=cnt, template=tmpl, slot_key=rho_cnt, nonce=nonce,
                    commitment=gamma, key_commitment=delta)
        self.slots[cnt] = slot
        self.stats["gen"] += 1
        return slot

    def gen_batch(self, counters: List[int]) -> List[Slot]:
        return [self.gen(c) for c in counters]

    # -- batch attestation (makes certificates transferable) ---------------
    def sign_batch(self, counters: List[int]) -> Tuple[bytes, bytes, MerkleTree]:
        items = [self.slots[c].leaf(self.ctx.sid) for c in counters]
        tree = MerkleTree.build(items)
        msg = batch_message(self.ctx.sid, counters, tree.root)
        self.stats["sign"] += 1
        return tree.root, self.attest.sign(msg), tree

    # -- OPEN (audit) ------------------------------------------------------
    def open_slot(self, cnt: int) -> Tuple[Template, bytes, bytes]:
        """Sacrifice a slot: reveal (template witness, nonce, rho_cnt)."""
        slot = self.slots[cnt]
        slot.state = SlotState.SACRIFICED
        self.stats["opened"] += 1
        w = slot.template.witness_bytes()
        self.stats["open_bytes"] += len(w) + len(slot.nonce) + len(slot.slot_key)
        return slot.template, slot.nonce, slot.slot_key

    def open_batch(self, marked: List[int], chi: bytes) -> "OpeningBundle":
        """Sacrifice every marked slot and *authenticate the whole opening*.

        The chip signs one Merkle root over all its openings for this audit
        round.  Without that signature a malicious host could corrupt an
        honest chip's opening and manufacture a certificate against it, so
        this single signature is exactly what buys defamation freeness; it is
        amortised over the whole audit round, not paid per slot.
        """
        openings: Dict[int, Opening] = {}
        leaves: List[bytes] = []
        for cnt in marked:
            if self.slots[cnt].state is not SlotState.READY:
                continue
            tmpl, nonce, rho_cnt = self.open_slot(cnt)
            op = Opening(cnt=cnt, slot_key=rho_cnt, nonce=nonce,
                         witness=tmpl.witness_bytes(), template=tmpl)
            openings[cnt] = op
            leaves.append(op.leaf(self.ctx.sid))
        if not leaves:
            return OpeningBundle(chi=chi, openings={}, order=[], root=b"",
                                 sig=b"", tree=None)
        tree = MerkleTree.build(leaves)
        msg = opening_message(self.ctx.sid, chi, len(leaves), tree.root)
        self.stats["sign"] += 1
        return OpeningBundle(chi=chi, openings=openings, order=list(openings.keys()),
                             root=tree.root, sig=self.attest.sign(msg), tree=tree)

    # -- BIND --------------------------------------------------------------
    def _bind_pads(self, cnt: int) -> Tuple[int, int]:
        p = self.ctx.params
        mask = kdf(self.slots[cnt].slot_key, b"PACT|bindmask", cnt, 16)
        return (int.from_bytes(mask[:8], "big") % p.domain,
                int.from_bytes(mask[8:], "big") % p.modulus)

    def bind_message(self, cnt: int, alpha_b: int, beta_b: int,
                     ic_commit_bytes: Optional[bytes] = None) -> bytes:
        """The single outbound BIND message (host-relayed, confidential, AE).

        Confidentiality: the body is encrypted under the channel
        encryption key (AES-CTR, nonce from the slot subkey and party bit) and
        the ciphertext is CMAC-authenticated under the derived MAC key, so a
        malicious relay host learns nothing from the message.  The two masked
        shares therefore cannot be summed to recover the input.
        """
        p = self.ctx.params
        ua, ub = self._bind_pads(cnt)
        sign = 1 if self.b == 0 else -1
        ma = (alpha_b + sign * ua) % p.domain
        mb = (beta_b + sign * ub) % p.modulus
        body = (struct.pack(">Q", cnt)
                + bytes([self.b])
                + ma.to_bytes((p.n + 7) // 8, "big")
                + mb.to_bytes((p.m + 7) // 8, "big")
                + (ic_commit_bytes or b""))
        nonce = kdf(self.slots[cnt].slot_key, b"PACT|bindnonce", cnt * 2 + self.b, 16)
        ct = channel_encrypt(self.ctx.channel_enc_key, nonce, body)
        framed = nonce + ct
        msg = framed + channel_mac(self.ctx.channel_mac_key, b"PACT|bind" + framed)
        self.stats["bind_bytes"] += len(msg)
        return msg

    def bind_finish(self, cnt: int, alpha_b: int, beta_b: int, peer_msg: bytes,
                    my_ic_opening: Optional[Tuple[int, int, bytes]] = None,
                    my_ic_commit: Optional[bytes] = None) -> Tuple[int, DPFKey]:
        """Reconstruct the input inside the boundary, patch, and emit a key.

        Returns ``(Delta, key_share)``.  ``Delta`` is public; the key share is
        a standard DPF key that EVAL consumes with no further interaction.

        Raises ``InputConsistencyAbort`` if the host bound a share other than
        the one it committed at submit time.
        """
        p = self.ctx.params
        slot = self.slots[cnt]
        if slot.state is not SlotState.READY:
            raise ProtocolAbort(f"slot {cnt} not READY (state={slot.state})")

        # --- input-consistency check against the host's submit-time commitment
        if my_ic_commit is not None:
            assert my_ic_opening is not None
            a_o, b_o, nu = my_ic_opening
            if (ic_commit(self.ctx.sid, cnt, self.b, a_o, b_o, nu, p) != my_ic_commit
                    or a_o != alpha_b or b_o != beta_b):
                self.stats["aborts"] += 1
                raise InputConsistencyAbort(self.b, cnt)

        nbytes = (p.n + 7) // 8
        mbytes = (p.m + 7) // 8
        # --- The peer message is nonce || ciphertext || MAC; the
        #     chip authenticates, then decrypts *before* any reconstruction.
        nonce = peer_msg[:16]
        ct, tag = peer_msg[16:-16], peer_msg[-16:]
        if not channel_mac_verify(self.ctx.channel_mac_key,
                                  b"PACT|bind" + nonce + ct, tag):
            raise ProtocolAbort("bind MAC failure (active network)")
        body = channel_decrypt(self.ctx.channel_enc_key, nonce, ct)
        pcnt = struct.unpack(">Q", body[:8])[0]
        pb = body[8]
        if pcnt != cnt or pb == self.b:
            raise ProtocolAbort("counter/party mismatch (replay)")
        off = 9
        pma = int.from_bytes(body[off:off + nbytes], "big"); off += nbytes
        pmb = int.from_bytes(body[off:off + mbytes], "big"); off += mbytes

        ua, ub = self._bind_pads(cnt)
        sign = 1 if self.b == 0 else -1
        my_ma = (alpha_b + sign * ua) % p.domain
        my_mb = (beta_b + sign * ub) % p.modulus

        alpha = (my_ma + pma) % p.domain      # the derived pads cancel
        beta = (my_mb + pmb) % p.modulus
        # --- start of the exposure window: the ONLY point in the whole protocol
        #     at which a reconstructed input exists inside the boundary
        from .primitives import aes_call_count
        aes_at_entry = aes_call_count()
        self._hbind = (alpha, beta)           # transient region H_bind

        delta = (slot.template.r - alpha) % p.domain
        key = slot.template.key_share(self.b, beta)   # THE single affine patch

        self._hbind = None                    # erased at commit
        # --- end of the exposure window
        self.stats["exposure_aes"] += aes_call_count() - aes_at_entry
        self.stats["exposure_ops"] += 2
        self._live_root = ratchet(self._live_root)
        slot.state = SlotState.USED
        self.stats["bind"] += 1
        return delta, key


@dataclass
class Opening:
    """One sacrificed slot, as revealed by the chip."""
    cnt: int
    slot_key: bytes
    nonce: bytes
    witness: bytes
    template: "Template"

    def leaf(self, sid: bytes) -> bytes:
        return h(b"PACT|open", sid, struct.pack(">Q", self.cnt),
                 self.slot_key, self.nonce, self.witness)

    @property
    def size_bytes(self) -> int:
        return len(self.slot_key) + len(self.nonce) + len(self.witness) + 8


def opening_leaf(sid: bytes, cnt: int, slot_key_b: bytes, nonce: bytes,
                 witness: bytes) -> bytes:
    return h(b"PACT|open", sid, struct.pack(">Q", cnt), slot_key_b, nonce, witness)


def opening_message(sid: bytes, chi: bytes, n: int, root: bytes) -> bytes:
    return (b"PACT|openbatch|" + sid + chi + struct.pack(">I", n) + root)


@dataclass
class OpeningBundle:
    chi: bytes
    openings: Dict[int, Opening]
    order: List[int]
    root: bytes
    sig: bytes
    tree: Optional[MerkleTree]

    def index_of(self, cnt: int) -> int:
        return self.order.index(cnt)

    @property
    def wire_bytes(self) -> int:
        return (sum(o.size_bytes for o in self.openings.values())
                + len(self.root) + len(self.sig))


def batch_message(sid: bytes, counters: List[int], root: bytes) -> bytes:
    return (b"PACT|batch|" + sid + struct.pack(">I", len(counters))
            + struct.pack(">Q", counters[0]) + root)


class ProtocolAbort(Exception):
    pass


class InputConsistencyAbort(ProtocolAbort):
    def __init__(self, party: int, cnt: int):
        super().__init__(f"input-consistency violation by party {party} on slot {cnt}")
        self.party = party
        self.cnt = cnt


# ---------------------------------------------------------------------------
# Input-consistency commitment (constrains the *host*, symmetric to the audit
# which constrains the *chip*)
# ---------------------------------------------------------------------------


def ic_commit(sid: bytes, cnt: int, party: int, alpha_b: int, beta_b: int,
              nu: bytes, p: DPFParams) -> bytes:
    return h(b"PACT|ic", sid, struct.pack(">QB", cnt, party),
             alpha_b.to_bytes((p.n + 7) // 8, "big"),
             beta_b.to_bytes((p.m + 7) // 8, "big"), nu)


@dataclass
class Host:
    """An untrusted host.  ``cheat_bind`` models the malicious-host adversary
    that commits one share at submit time and binds a different one."""
    b: int
    ctx: PairingContext
    rng: DeterministicRNG
    cheat_bind: bool = False

    def submit(self, cnt: int, alpha_b: int, beta_b: int
               ) -> Tuple[bytes, Tuple[int, int, bytes]]:
        nu = self.rng.bytes(16)
        cm = ic_commit(self.ctx.sid, cnt, self.b, alpha_b, beta_b, nu, self.ctx.params)
        return cm, (alpha_b, beta_b, nu)

    def bind_inputs(self, opening: Tuple[int, int, bytes]) -> Tuple[int, int]:
        a, b, _ = opening
        if self.cheat_bind:
            return (a + 1) % self.ctx.params.domain, b
        return a, b


# ---------------------------------------------------------------------------
# End-to-end honest execution
# ---------------------------------------------------------------------------


def run_keygen(chip0: Chip, chip1: Chip, host0: Host, host1: Host,
               cnt: int, alpha: int, beta: int, rng: DeterministicRNG
               ) -> Tuple[int, DPFKey, DPFKey]:
    """One full BIND over a pre-generated slot.  Returns (Delta, k0, k1)."""
    p = chip0.ctx.params
    a0 = rng.randint(p.domain)
    a1 = (alpha - a0) % p.domain
    b0 = rng.randint(p.modulus)
    b1 = (beta - b0) % p.modulus

    cm0, op0 = host0.submit(cnt, a0, b0)
    cm1, op1 = host1.submit(cnt, a1, b1)

    ia0, ib0 = host0.bind_inputs(op0)
    ia1, ib1 = host1.bind_inputs(op1)

    m0 = chip0.bind_message(cnt, ia0, ib0, cm0)
    m1 = chip1.bind_message(cnt, ia1, ib1, cm1)

    d0, k0 = chip0.bind_finish(cnt, ia0, ib0, m1,
                               my_ic_opening=(ia0, ib0, op0[2]), my_ic_commit=cm0)
    d1, k1 = chip1.bind_finish(cnt, ia1, ib1, m0,
                               my_ic_opening=(ia1, ib1, op1[2]), my_ic_commit=cm1)
    assert d0 == d1, "chips must agree on the public shift"
    return d0, k0, k1


def eval_key(key: DPFKey, delta: int, x: int) -> int:
    """EVAL: local evaluation of the bound key at a public input x."""
    return dpf_eval(key, (x + delta) % key.params.domain)
