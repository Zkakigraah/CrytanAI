"""
Atbash Cipher Solver.

There's no key to search for -- Atbash has exactly one possible mapping
(the alphabet reversed) and is its own inverse. "Solving" it is really
just running the transform and reporting how English-like the result is;
this file exists mainly so Atbash fits the same uniform solver interface
(`list[SolveResult]`) as every other cipher, which matters for the
detector and the UI, both of which call solvers generically.
"""
from ciphers.atbash import decrypt_atbash
from ml.nlp_scorer import default_scorer
from solvers.base import SolveResult


def solve_atbash(ciphertext: str, top_k: int = 3) -> list[SolveResult]:
    """Returns the single Atbash decryption (there is only one possible
    key), wrapped in the same list[SolveResult] shape every other solver
    uses."""
    plaintext = decrypt_atbash(ciphertext)
    result = SolveResult(
        key=None, plaintext=plaintext, score=default_scorer.score(plaintext),
        method="direct", confidence="high",
    )
    return [result][:top_k]
