"""
Caesar Cipher Solver.

Brute force: only 26 possible shifts, so trying every single one and
letting the English scorer rank them is exact and instant -- no search
heuristics needed. This is the simplest possible instance of the whole
project's core pattern (guess a key, decrypt, score, keep the best), and
a good place to see that pattern with nothing else in the way.
"""
from ciphers.caesar import decrypt_caesar
from solvers.base import SolveResult, brute_force_rank


def solve_caesar(ciphertext: str, top_k: int = 3) -> list[SolveResult]:
    """Returns the top_k most English-like Caesar decryptions, ranked
    best-first, out of all 26 possible shifts."""
    return brute_force_rank(
        ciphertext,
        candidate_keys=range(26),
        decrypt_fn=decrypt_caesar,
        method="brute_force",
        confidence="high",
        top_k=top_k,
    )
