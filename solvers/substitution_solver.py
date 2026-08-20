"""
Substitution Cipher Solver.

General monoalphabetic substitution has 26! (~4x10^26) possible keys --
far too many to brute force, unlike Caesar's 26 or Affine's 312. The
standard technique (used by essentially every serious substitution-cipher
solver) is hill-climbing: start from a plausible guess, repeatedly swap
two letters in the candidate key, and keep the swap if it raises the
decryption's score under the English quadgram model. A single run can get
stuck at a key that's locally un-improvable by one swap but still wrong,
so this runs many independent random restarts and keeps the best result
across all of them.

This same search also solves ciphers/keyword.py's output: ciphertext-only,
a keyword-derived substitution alphabet is statistically indistinguishable
from a randomly-shuffled one (both are "some permutation of the
alphabet"), so there's no separate keyword solver -- this one covers both.
"""
import random
from collections import Counter

from ciphers.substitution import decrypt_substitution
from ml.nlp_scorer import default_scorer
from solvers.base import SolveResult
from utils.text_utils import ALPHABET, index_of_coincidence

RESTARTS = 25
ITERATIONS_PER_RESTART = 1500

# Classic ETAOIN SHRDLU ordering -- English letters, most to least frequent.
ENGLISH_FREQ_ORDER = "ETAOINSHRDLCUMWFGYPBVKJXQZ"


def _frequency_seeded_key(ciphertext: str) -> str:
    """A smarter starting point than a pure random shuffle: guess that
    the most frequent ciphertext letters decrypt to the most frequent
    English letters. This doesn't need to be correct -- hill-climbing
    will fix individual wrong guesses -- it just needs to be a better
    starting point than chance, which measurably speeds up convergence."""
    letters_only = [c for c in ciphertext.upper() if c in ALPHABET]
    cipher_by_freq = [ch for ch, _ in Counter(letters_only).most_common()]
    for ch in ALPHABET:  # letters absent from the ciphertext still need a slot
        if ch not in cipher_by_freq:
            cipher_by_freq.append(ch)

    plain_to_cipher = dict(zip(ENGLISH_FREQ_ORDER, cipher_by_freq))
    return "".join(plain_to_cipher[ch] for ch in ALPHABET)


def _hill_climb(ciphertext: str, start_key: str, iterations: int) -> str:
    """Local search over 26-letter keys via random pairwise swaps,
    scored by quadgram log-likelihood (fast -- this runs thousands of
    times per restart, so it uses quick_score, not the full multi-signal
    score). Returns the best key found in this run."""
    current_key = list(start_key)
    current_score = default_scorer.quick_score(decrypt_substitution(ciphertext, start_key))
    best_key, best_score = "".join(current_key), current_score

    for _ in range(iterations):
        i, j = random.sample(range(26), 2)
        current_key[i], current_key[j] = current_key[j], current_key[i]

        candidate = "".join(current_key)
        score = default_scorer.quick_score(decrypt_substitution(ciphertext, candidate))

        if score >= current_score:
            current_score = score
            if score > best_score:
                best_key, best_score = candidate, score
        else:
            current_key[i], current_key[j] = current_key[j], current_key[i]  # undo

    return best_key


def solve_substitution(ciphertext: str, top_k: int = 3) -> list[SolveResult]:
    """
    Hill-climbs the substitution key over many random restarts and
    returns the top_k distinct-plaintext results found, ranked by the
    full English score.
    """
    # A monoalphabetic substitution preserves the ciphertext's index of
    # coincidence at roughly the English value (~0.067); a markedly lower
    # IC is more consistent with a polyalphabetic cipher. Still worth
    # attempting -- just flagged as lower-confidence if so.
    confidence = "high" if index_of_coincidence(ciphertext) >= 0.045 else "medium"

    seeds = [_frequency_seeded_key(ciphertext)]
    seeds += ["".join(random.sample(ALPHABET, 26)) for _ in range(RESTARTS - 1)]

    found: dict[str, str] = {}  # plaintext -> key
    for seed in seeds:
        key = _hill_climb(ciphertext, seed, ITERATIONS_PER_RESTART)
        plaintext = decrypt_substitution(ciphertext, key)
        found[plaintext] = key

    scored = [(pt, key, default_scorer.score(pt)) for pt, key in found.items()]
    scored.sort(key=lambda t: -t[2])

    return [
        SolveResult(key=key, plaintext=pt, score=score, method="hill_climbing", confidence=confidence)
        for pt, key, score in scored[:top_k]
    ]
