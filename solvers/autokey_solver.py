"""
Autokey Cipher Solver.

Autokey defeats the standard Vigenere attack by design: because the
keystream is primer + plaintext, it (almost) never repeats, so Kasiski
examination and the IC-scan technique that finds Vigenere/Beaufort/
Gronsfeld's key length (see vigenere_family_solver.py) find nothing
meaningful here -- the effective key length is close to the message
length itself.

What's still attackable is the PRIMER, which is typically short (a few
letters) precisely because it has to be memorable. This searches over
primer length and content directly: exhaustively for very short primers
(3 letters or fewer -- 26^3 = 17,576 candidates, still fast), and via
hill-climbing for longer ones. Because a wrong primer guess decodes
correctly only up to the point it diverges from the true primer and then
cascades into garbage from there (the self-keying recursion means one
wrong recovered letter corrupts every keystream position depending on
it), this attack is inherently less reliable than the closed-form ones
elsewhere in this project -- every result here is reported as
best_effort confidence, regardless of primer length or how clean the
decryption looks.
"""
import random
from itertools import product

from ciphers.autokey import decrypt_autokey
from ml.nlp_scorer import default_scorer
from solvers.base import SolveResult
from utils.text_utils import ALPHABET

MAX_PRIMER_LENGTH = 6
EXHAUSTIVE_PRIMER_LIMIT = 3
HILL_CLIMB_RESTARTS = 10
HILL_CLIMB_ITERATIONS = 200


def _exhaustive_primer(ciphertext: str, length: int) -> str:
    best_primer, best_score = None, float("-inf")
    for combo in product(ALPHABET, repeat=length):
        primer = "".join(combo)
        score = default_scorer.quick_score(decrypt_autokey(ciphertext, primer))
        if score > best_score:
            best_score, best_primer = score, primer
    return best_primer


def _hill_climb_primer(ciphertext: str, length: int) -> str:
    best_overall, best_overall_score = None, float("-inf")
    for _ in range(HILL_CLIMB_RESTARTS):
        primer = [random.choice(ALPHABET) for _ in range(length)]
        score = default_scorer.quick_score(decrypt_autokey(ciphertext, "".join(primer)))

        for _ in range(HILL_CLIMB_ITERATIONS):
            pos = random.randrange(length)
            original = primer[pos]
            primer[pos] = random.choice(ALPHABET)
            new_score = default_scorer.quick_score(decrypt_autokey(ciphertext, "".join(primer)))
            if new_score >= score:
                score = new_score
            else:
                primer[pos] = original  # undo

        if score > best_overall_score:
            best_overall_score, best_overall = score, "".join(primer)

    return best_overall


def solve_autokey(ciphertext: str, top_k: int = 3) -> list[SolveResult]:
    """Searches primer length 1..MAX_PRIMER_LENGTH -- exhaustively for
    short primers, via hill-climbing for longer ones -- and returns the
    top_k results, always at best_effort confidence (see module
    docstring)."""
    found: dict[str, tuple[str, float]] = {}
    for length in range(1, MAX_PRIMER_LENGTH + 1):
        if length <= EXHAUSTIVE_PRIMER_LIMIT:
            primer = _exhaustive_primer(ciphertext, length)
        else:
            primer = _hill_climb_primer(ciphertext, length)

        plaintext = decrypt_autokey(ciphertext, primer)
        score = default_scorer.score(plaintext)
        if plaintext not in found or score > found[plaintext][1]:
            found[plaintext] = (primer, score)

    ranked = sorted(found.items(), key=lambda kv: -kv[1][1])
    return [
        SolveResult(key=primer, plaintext=pt, score=score, method="primer_search", confidence="best_effort")
        for pt, (primer, score) in ranked[:top_k]
    ]
