"""
PACT reference implementation -- cryptographic primitives.

Everything the fixed-function PACT chip needs is built from a *single* AES-128
core, exactly as the hardware datapath described in the paper: the length-
doubling PRG (fixed-key AES in Matyas--Meyer--Oseas mode), the counter-bound
slot commitment (an MMO hash on the same core), the channel MAC and mask
derivation all time-share one iterative AES-128 block.  The only asymmetric
primitive is the attestation signature (Ed25519), which the chip already
provisions for pairing and which PACT reuses once per audit batch so that a
cheating certificate is *transferable*.

Design notes that matter for the paper:

  * ``prg`` is the GGM length-doubling expander G : {0,1}^128 -> {0,1}^256.
    We use two fixed, publicly known AES keys in MMO mode
    (G(s) = (AES_{K0}(s) XOR s) || (AES_{K1}(s) XOR s)).  This is the standard
    fixed-key blockcipher instantiation of a correlation-robust hash
    (Bellare et al., IEEE S&P 2013) and is what a secure element can afford.

  * ``slot_commit`` is the counter-bound commitment gamma_cnt: a *public*
    hash commitment, hiding through a per-slot nonce and binding through
    collision resistance.  It is deliberately not a MAC -- a third party must
    be able to check it, or a certificate would not be transferable.

  * ``slot_key`` implements the two-level derivation rho -> rho_cnt that makes
    public verifiability possible: opening a *sacrificed* slot reveals only
    that slot's subkey, which is enough for anyone to recompute the template
    and convict, and useless for anything else.

  * ``MerkleTree`` gives O(log N) certificate size: a certificate names one
    slot, not the whole batch.

All functions are deterministic given their inputs; every source of randomness
in the whole codebase flows from an explicit seed so that every number in the
paper is reproducible.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import struct
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.cmac import CMAC
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives import serialization
from cryptography.exceptions import InvalidSignature

LAMBDA_BYTES = 16  # 128-bit seeds; matches the AES-128 datapath
LAMBDA_BITS = 128

# ---------------------------------------------------------------------------
# Fixed-key AES, used as the chip's PRG core.
# ---------------------------------------------------------------------------

# Two publicly known, fixed AES keys.  Nothing secret lives here; the security
# of the GGM expansion comes from the secret seed, not from these constants.
_PRG_KEY_0 = bytes.fromhex("00112233445566778899aabbccddeeff")
_PRG_KEY_1 = bytes.fromhex("0f1e2d3c4b5a69788796a5b4c3d2e1f0")

_AES_CALLS = 0  # instrumentation: counts block-cipher invocations


def _aes_ecb_encrypt(key: bytes, block: bytes) -> bytes:
    enc = Cipher(algorithms.AES(key), modes.ECB()).encryptor()
    return enc.update(block) + enc.finalize()


def _xor(a: bytes, b: bytes) -> bytes:
    return bytes(x ^ y for x, y in zip(a, b))


def aes_call_count() -> int:
    """Number of AES block invocations issued so far (cost instrumentation)."""
    return _AES_CALLS


def reset_aes_call_count() -> None:
    global _AES_CALLS
    _AES_CALLS = 0


def prg(seed: bytes) -> Tuple[bytes, int, bytes, int]:
    """Length-doubling PRG G: {0,1}^128 -> {0,1}^{2*(128+1)}.

    Returns (s_left, t_left, s_right, t_right).  The control bits t are the
    least-significant bit of each half; the seed halves keep all 128 bits
    (the low bit is reused as the control bit, which is the standard
    space-saving convention of Boyle et al. and costs nothing in security
    because the PRG output is pseudorandom).
    """
    global _AES_CALLS
    assert len(seed) == LAMBDA_BYTES
    left = _xor(_aes_ecb_encrypt(_PRG_KEY_0, seed), seed)
    right = _xor(_aes_ecb_encrypt(_PRG_KEY_1, seed), seed)
    _AES_CALLS += 2
    return left, left[-1] & 1, right, right[-1] & 1


# ---------------------------------------------------------------------------
# Key-derivation / mask derivation (also on the AES core in hardware; modelled
# here with HMAC-SHA256 for clarity of domain separation).
# ---------------------------------------------------------------------------


def kdf(root: bytes, label: bytes, counter: int, nbytes: int) -> bytes:
    """Domain-separated derivation from the pairing root rho.

    ``Derive(rho, cnt)`` in the paper is a call to this function with the
    appropriate label.  It is deterministic, so both paired chips derive the
    same template with zero communication.
    """
    out = b""
    block = 0
    while len(out) < nbytes:
        out += hmac.new(
            root,
            label + b"|" + struct.pack(">Q", counter) + b"|" + struct.pack(">I", block),
            hashlib.sha256,
        ).digest()
        block += 1
    return out[:nbytes]


def ratchet(root: bytes) -> bytes:
    """One-way root ratchet (forward secrecy for consumed slots)."""
    return hashlib.sha256(b"PACT|ratchet|" + root).digest()[:32]


def slot_key(root: bytes, cnt: int) -> bytes:
    """Per-slot derivation subkey rho_cnt = KDF(rho, "slotkey", cnt).

    Two-level derivation is what makes a cheating certificate *transferable*.
    Opening a sacrificed slot reveals rho_cnt, never rho: a third party can
    recompute that one template from rho_cnt and convict, while PRF security
    keeps rho and every other slot's subkey hidden.  A sacrificed slot is by
    definition never used for a real key, so rho_cnt is worthless to an
    attacker.
    """
    return kdf(root, b"PACT|slotkey", cnt, 32)


def slot_key_commit(slot_key_bytes: bytes) -> bytes:
    """Public commitment delta_cnt = H(rho_cnt), published inside the batch."""
    return h(b"PACT|slotkeycom", slot_key_bytes)


# ---------------------------------------------------------------------------
# AES-MMO compression -- the on-chip instantiation of the CRHF used for the
# slot commitment.  The reference implementation uses SHA-256 (``h``) for
# clarity; this function documents, and exercises, the claim that the same
# single AES-128 core realises the commitment datapath.
# ---------------------------------------------------------------------------


def aes_mmo_hash(data: bytes) -> bytes:
    """Matyas--Meyer--Oseas hash over the chip's AES-128 core (128-bit output)."""
    global _AES_CALLS
    pad = (data + b"\x80" + b"\x00" * ((-(len(data) + 9)) % 16)
           + struct.pack(">Q", len(data)))
    state = b"\x00" * 16
    for i in range(0, len(pad), 16):
        blk = pad[i:i + 16]
        state = _xor(_aes_ecb_encrypt(state, blk), blk)
        _AES_CALLS += 1
    return state


# ---------------------------------------------------------------------------
# Hashing / digests
# ---------------------------------------------------------------------------


def h(*parts: bytes) -> bytes:
    """Domain-separated collision-resistant hash (SHA-256, 32 bytes)."""
    ctx = hashlib.sha256()
    for p in parts:
        ctx.update(struct.pack(">I", len(p)))
        ctx.update(p)
    return ctx.digest()


def slot_digest(witness_bytes: bytes, cnt: int) -> bytes:
    """Digest of a template slot, bound to its monotone counter."""
    return h(b"PACT|slotdigest", struct.pack(">Q", cnt), witness_bytes)


# ---------------------------------------------------------------------------
# Counter-bound commitment  gamma_cnt = Com(K_ch, cnt || slot)
# ---------------------------------------------------------------------------


def slot_commit(sid: bytes, cnt: int, digest: bytes, nonce: bytes) -> bytes:
    """Counter-bound commitment gamma_cnt = Com(sid, cnt, slot).

    A *public* hash commitment, not a MAC: a third party must be able to check
    it, which is what makes a cheating certificate transferable.  It is hiding
    because of the per-slot nonce (a slot digest is a deterministic function
    of (rho,cnt), so an unnonced commitment would be testable by anyone who
    guessed rho) and binding by collision resistance.  It is bound to the
    session id, so a commitment cannot be replayed across pairings.

    On the chip this CRHF is the AES-128 core in MMO mode (``aes_mmo_hash``);
    the reference implementation uses SHA-256 for clarity.
    """
    return h(b"PACT|com", sid, struct.pack(">Q", cnt), nonce, digest)


def slot_commit_verify(sid: bytes, cnt: int, digest: bytes, nonce: bytes, tag: bytes) -> bool:
    return hmac.compare_digest(slot_commit(sid, cnt, digest, nonce), tag)


def channel_mac(channel_key: bytes, msg: bytes) -> bytes:
    """Authenticated inter-chip channel tag (CMAC-AES-128 on the same core)."""
    global _AES_CALLS
    c = CMAC(algorithms.AES(channel_key))
    c.update(msg)
    _AES_CALLS += (len(msg) + 15) // 16 + 1
    return c.finalize()


def channel_mac_verify(channel_key: bytes, msg: bytes, tag: bytes) -> bool:
    return hmac.compare_digest(channel_mac(channel_key, msg), tag)


# ---------------------------------------------------------------------------
# Confidential inter-chip channel: encrypt-then-MAC on the AES core.
# ---------------------------------------------------------------------------
# BIND messages are confidentiality protected: a malicious host relays both
# messages but cannot read them; the two masked shares therefore cannot be
# summed by the relay to recover the secret input alpha.  The channel uses
# AES-128-CTR encryption under an encryption key derived from K_ch,
# authenticated with CMAC under a separately derived key, and a per-message
# nonce derived from the slot subkey and the sender's party bit.  Everything
# runs on the same single AES-128 core (CTR keystream + CMAC).


def derive_channel_keys(channel_key: bytes) -> Tuple[bytes, bytes]:
    """(encryption key, MAC key) of the confidential inter-chip channel."""
    return (kdf(channel_key, b"PACT|chenc", 0, 16),
            kdf(channel_key, b"PACT|chmac", 0, 16))


def channel_encrypt(enc_key: bytes, nonce: bytes, body: bytes) -> bytes:
    """AES-128-CTR encryption of a BIND body under the derived channel key."""
    global _AES_CALLS
    enc = Cipher(algorithms.AES(enc_key), modes.CTR(nonce)).encryptor()
    ct = enc.update(body) + enc.finalize()
    _AES_CALLS += (len(body) + 15) // 16
    return ct


def channel_decrypt(enc_key: bytes, nonce: bytes, ct: bytes) -> bytes:
    """AES-128-CTR decryption (same keystream; counted as block operations)."""
    global _AES_CALLS
    dec = Cipher(algorithms.AES(enc_key), modes.CTR(nonce)).decryptor()
    body = dec.update(ct) + dec.finalize()
    _AES_CALLS += (len(ct) + 15) // 16
    return body


# ---------------------------------------------------------------------------
# Merkle tree over a batch of slot commitments (transferable certificates)
# ---------------------------------------------------------------------------


def _mleaf(x: bytes) -> bytes:
    return h(b"PACT|mleaf", x)


def _mnode(a: bytes, b: bytes) -> bytes:
    return h(b"PACT|mnode", a, b)


@dataclass
class MerkleTree:
    leaves: List[bytes]
    levels: List[List[bytes]]

    @classmethod
    def build(cls, items: Sequence[bytes]) -> "MerkleTree":
        assert len(items) > 0
        level = [_mleaf(x) for x in items]
        levels = [level]
        while len(level) > 1:
            nxt = []
            for i in range(0, len(level), 2):
                a = level[i]
                b = level[i + 1] if i + 1 < len(level) else level[i]
                nxt.append(_mnode(a, b))
            levels.append(nxt)
            level = nxt
        return cls(leaves=list(items), levels=levels)

    @property
    def root(self) -> bytes:
        return self.levels[-1][0]

    def path(self, index: int) -> List[Tuple[int, bytes]]:
        """Authentication path: list of (side, sibling) from leaf to root."""
        out: List[Tuple[int, bytes]] = []
        idx = index
        for level in self.levels[:-1]:
            sib_idx = idx ^ 1
            sib = level[sib_idx] if sib_idx < len(level) else level[idx]
            out.append((idx & 1, sib))
            idx >>= 1
        return out


def merkle_verify(item: bytes, index: int, path: Sequence[Tuple[int, bytes]], root: bytes) -> bool:
    cur = _mleaf(item)
    idx = index
    for side, sib in path:
        if side == 0:
            cur = _mnode(cur, sib)
        else:
            cur = _mnode(sib, cur)
        idx >>= 1
    return hmac.compare_digest(cur, root)


# ---------------------------------------------------------------------------
# Attestation signatures (already provisioned on-die for pairing; PACT reuses
# them once per audit batch so that certificates are publicly verifiable)
# ---------------------------------------------------------------------------


@dataclass
class AttestationKey:
    sk: Ed25519PrivateKey

    @classmethod
    def generate(cls, rng: "DeterministicRNG") -> "AttestationKey":
        return cls(sk=Ed25519PrivateKey.from_private_bytes(rng.bytes(32)))

    @property
    def public_bytes(self) -> bytes:
        return self.sk.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )

    def sign(self, msg: bytes) -> bytes:
        return self.sk.sign(msg)


def attest_verify(public_bytes: bytes, msg: bytes, sig: bytes) -> bool:
    try:
        Ed25519PublicKey.from_public_bytes(public_bytes).verify(sig, msg)
        return True
    except (InvalidSignature, ValueError):
        return False


# ---------------------------------------------------------------------------
# Deterministic randomness (every experiment is seed-reproducible)
# ---------------------------------------------------------------------------


class DeterministicRNG:
    """A seekable, seed-reproducible byte stream (SHA-256 in counter mode)."""

    def __init__(self, seed, label: bytes = b"PACT"):
        if isinstance(seed, int):
            seed = struct.pack(">Q", seed)
        elif isinstance(seed, str):
            seed = seed.encode()
        self._seed = hashlib.sha256(label + b"|" + seed).digest()
        self._ctr = 0
        self._buf = b""

    def bytes(self, n: int) -> bytes:
        while len(self._buf) < n:
            self._buf += hashlib.sha256(self._seed + struct.pack(">Q", self._ctr)).digest()
            self._ctr += 1
        out, self._buf = self._buf[:n], self._buf[n:]
        return out

    def randint(self, bound: int) -> int:
        """Uniform in [0, bound) via rejection sampling (unbiased)."""
        assert bound > 0
        nbits = bound.bit_length()
        nbytes = (nbits + 7) // 8
        mask = (1 << nbits) - 1
        while True:
            v = int.from_bytes(self.bytes(nbytes), "big") & mask
            if v < bound:
                return v

    def random(self) -> float:
        """Uniform in [0,1) with 53 bits of precision."""
        return int.from_bytes(self.bytes(8), "big") / 2.0**64

    def bernoulli(self, p: float) -> bool:
        return self.random() < p


def urandom(n: int) -> bytes:
    return os.urandom(n)
