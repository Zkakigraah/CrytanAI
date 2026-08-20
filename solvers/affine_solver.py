"""
Affine Cipher Solver.

Brute force over all valid (a, b) pairs: a must be coprime with 26, which
leaves exactly 12 valid values (1, 3, 5, 7, 9, 11, 15, 17, 19, 21, 23, 25)
times 26 possible values of b = 312 total keys. Still small enough to try
every single one exactly, same as Caesar (which is just this cipher's
a=1 special case).
"""
import math

from ciphers.affine import decrypt_affine
from solvers.base import SolveResult, brute_force_rank

VALID_A = [a for a in range(1, 26) if math.gcd(a, 26) == 1]
ALL_KEYS = [(a, b) for a in VALID_A for b in range(26)]


def solve_affine(ciphertext: str, top_k: int = 3) -> list[SolveResult]:
    """Returns the top_k most English-like Affine decryptions, ranked
    best-first, out of all 312 valid (a, b) keys."""
    return brute_force_rank(
        ciphertext,
        candidate_keys=ALL_KEYS,
        decrypt_fn=lambda text, key: decrypt_affine(text, key[0], key[1]),
        method="brute_force",
        confidence="high",
        top_k=top_k,
    )
