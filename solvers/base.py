"""
Shared infrastructure for solvers/. Every solver function returns a
ranked list of SolveResult -- this is what lets the UI show a top-3
ranking uniformly regardless of which cipher/algorithm produced it, and
what lets ml/cipher_detector.py cross-check its own prediction against a
solver's opinion if it wants to.
"""
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from ml.nlp_scorer import default_scorer


@dataclass
class SolveResult:
    key: Any            # the recovered key, in whatever shape that cipher uses
    plaintext: str       # the candidate decryption
    score: float          # full_score() of the plaintext -- higher = more English-like
    method: str           # e.g. "brute_force", "hill_climbing", "kasiski+frequency"
    confidence: str       # "high" | "medium" | "best_effort" -- coarse, human-readable


def brute_force_rank(
    ciphertext: str,
    candidate_keys: Iterable[Any],
    decrypt_fn: Callable[[str, Any], str],
    method: str,
    confidence: str = "high",
    top_k: int = 3,
) -> list[SolveResult]:
    """
    Tries every key in `candidate_keys`, scores the resulting decryption,
    and returns the top_k highest-scoring results. This is the shared
    shape behind every brute-forceable solver (Caesar: 26 shifts, Affine:
    312 (a,b) pairs) -- small, closed key spaces where "try them all" is
    cheap and exact, no heuristic search required.
    """
    results = []
    for key in candidate_keys:
        plaintext = decrypt_fn(ciphertext, key)
        score = default_scorer.score(plaintext)
        results.append(SolveResult(key=key, plaintext=plaintext, score=score,
                                    method=method, confidence=confidence))

    results.sort(key=lambda r: -r.score)
    return results[:top_k]
