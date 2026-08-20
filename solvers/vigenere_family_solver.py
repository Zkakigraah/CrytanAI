"""
Vigenere-Family Solver.

Covers Vigenere, Beaufort, and Gronsfeld -- three ciphers that all work
the same way structurally (a short key cycles, one key-symbol shifting
each letter of plaintext) and so share the exact same attack, which is
real frequency analysis in two stages:

  1. Find the key LENGTH. A repeating key creates two independent
     signals: repeated ciphertext substrings tend to reappear at
     distances that are multiples of the key length (Kasiski
     examination), and splitting the ciphertext into that many
     interleaved columns produces columns whose Index of Coincidence
     looks like single-alphabet English (~0.067) rather than random
     (~0.0385) -- because each column, if the length guess is right, is
     really just Caesar-shifted English (this is the same idea behind
     the classical Friedman test, applied here by direct measurement
     across candidate lengths rather than a closed-form estimator).

  2. Recover the key, one symbol at a time, by frequency analysis. Once
     the length is fixed, each of the L columns is a simple single-shift
     cipher on its own -- exactly the Caesar-solving problem, repeated L
     times, using single-letter chi-squared fit (the classic frequency-
     analysis tool) rather than the full English scorer, since a column
     in isolation is too short and disordered for n-gram statistics to
     help.

This is exactly the technique that DOESN'T generalize to Hill (see
solvers/hill_solver.py): Hill's block/matrix structure doesn't leave
single-letter statistical residue behind the way a per-position shift
does, which is why Hill needs a fundamentally different (algebraic,
known-plaintext) attack instead.
"""
from collections import Counter
from typing import Callable

from ciphers.vigenere import decrypt_vigenere
from ciphers.beaufort import decrypt_beaufort
from ciphers.gronsfeld import decrypt_gronsfeld
from ml.nlp_scorer import default_scorer
from solvers.base import SolveResult
from utils.text_utils import ALPHABET, index_of_coincidence, clean_letters_only
from utils.language_stats import chi_squared_statistic

MAX_KEY_LENGTH = 20
KASISKI_NGRAM = 3


def _kasiski_candidate_lengths(letters: str) -> list[int]:
    """Finds repeated n-grams in the ciphertext and records the distances
    between repeats. The true key length is likely a common factor of
    those distances -- tally small factors across every observed
    distance and rank them by how often they show up."""
    positions: dict[str, list[int]] = {}
    for i in range(len(letters) - KASISKI_NGRAM + 1):
        gram = letters[i:i + KASISKI_NGRAM]
        positions.setdefault(gram, []).append(i)

    factor_votes = Counter()
    for idxs in positions.values():
        if len(idxs) < 2:
            continue
        for a, b in zip(idxs, idxs[1:]):
            distance = b - a
            for length in range(2, MAX_KEY_LENGTH + 1):
                if distance % length == 0:
                    factor_votes[length] += 1

    return [length for length, _ in factor_votes.most_common(6)]


def _ic_scan_candidate_lengths(letters: str) -> list[int]:
    """For each plausible key length, splits the ciphertext into that
    many interleaved columns and measures their average Index of
    Coincidence. Lengths whose columns look like single-alphabet English
    (high IC) are ranked first -- this is the Friedman test's underlying
    idea, applied by direct measurement."""
    scored = []
    for length in range(1, MAX_KEY_LENGTH + 1):
        columns = [letters[i::length] for i in range(length)]
        ics = [index_of_coincidence(col) for col in columns if len(col) >= 2]
        if ics:
            scored.append((length, sum(ics) / len(ics)))

    scored.sort(key=lambda t: -t[1])
    return [length for length, _ in scored[:6]]


def _best_shift_for_column(column: str, shift_range: range, column_decrypt: Callable) -> int:
    """Tries every shift in shift_range, decrypts the column with it,
    and returns the shift whose result has the lowest chi-squared
    distance from expected English letter frequencies -- ordinary single-
    alphabet frequency analysis, just scoped to one key position."""
    best_shift, best_chi2 = shift_range[0], float("inf")
    for shift in shift_range:
        chi2 = chi_squared_statistic(column_decrypt(column, shift))
        if chi2 < best_chi2:
            best_chi2, best_shift = chi2, shift
    return best_shift


def _minimal_period(key: str) -> str:
    """If key is a shorter unit repeated a whole number of times (which
    happens when a tried candidate length is a multiple of the true
    period -- both decrypt identically, so the search has no way to
    prefer the shorter one on correctness grounds alone), returns that
    shorter unit. Otherwise returns key unchanged. This keeps reported
    keys canonical instead of e.g. 'SECRETSECRETSECRET'."""
    n = len(key)
    for d in range(1, n):
        if n % d == 0 and key[:d] * (n // d) == key:
            return key[:d]
    return key


def _refine_key(
    ciphertext: str,
    key: str,
    decrypt_fn: Callable[[str, str], str],
    alphabet_symbols: str,
    sweeps: int = 2,
) -> str:
    """
    Polishes a frequency-analysis key guess. Per-column chi-squared can
    genuinely misfire when a column is short -- verified directly: an
    isolated 38-letter column can have a *lower* chi-squared distance
    under the wrong shift than the true one, just from sampling noise.
    But the full ciphertext's quadgram statistics are a much stronger
    signal once a starting key already has most positions right, so this
    polishes with the same coordinate-ascent idea the substitution solver
    uses: for each key position, try every possible replacement symbol,
    keep whichever change most improves the *whole* decryption's score,
    and repeat for a few sweeps until nothing improves.
    """
    key_chars = list(key)
    current_score = default_scorer.quick_score(decrypt_fn(ciphertext, "".join(key_chars)))

    for _ in range(sweeps):
        improved = False
        for pos in range(len(key_chars)):
            original = key_chars[pos]
            best_symbol, best_score = original, current_score
            for symbol in alphabet_symbols:
                if symbol == original:
                    continue
                key_chars[pos] = symbol
                score = default_scorer.quick_score(decrypt_fn(ciphertext, "".join(key_chars)))
                if score > best_score:
                    best_score, best_symbol = score, symbol
            key_chars[pos] = best_symbol
            if best_symbol != original:
                improved = True
                current_score = best_score
        if not improved:
            break

    return "".join(key_chars)


def _solve_family(
    ciphertext: str,
    decrypt_fn: Callable[[str, str], str],
    column_decrypt: Callable[[str, int], str],
    shift_range: range,
    key_symbol_fn: Callable[[int], str],
    alphabet_symbols: str,
    top_k: int,
) -> list[SolveResult]:
    letters = clean_letters_only(ciphertext)
    if len(letters) < 20:
        # Too short for Kasiski/IC-scan to say anything reliable --
        # just try every plausible short length directly.
        candidate_lengths = list(range(1, max(2, min(MAX_KEY_LENGTH, len(letters) // 2) + 1)))
    else:
        ic_scan = _ic_scan_candidate_lengths(letters)
        kasiski = _kasiski_candidate_lengths(letters)
        candidate_lengths = []
        for length in ic_scan + kasiski:  # IC-scan first: more reliable when text is short
            if length not in candidate_lengths:
                candidate_lengths.append(length)
        candidate_lengths = candidate_lengths[:8] or [1]

    found: dict[str, tuple[str, float]] = {}
    for length in candidate_lengths:
        columns = [letters[i::length] for i in range(length)]
        key = "".join(
            key_symbol_fn(_best_shift_for_column(col, shift_range, column_decrypt))
            for col in columns
        )
        key = _refine_key(ciphertext, key, decrypt_fn, alphabet_symbols)
        key = _minimal_period(key)

        plaintext = decrypt_fn(ciphertext, key)
        score = default_scorer.score(plaintext)
        if plaintext not in found or score > found[plaintext][1]:
            found[plaintext] = (key, score)

    ranked = sorted(found.items(), key=lambda kv: -kv[1][1])
    return [
        SolveResult(key=key, plaintext=pt, score=score,
                    method="kasiski+ic_scan+frequency", confidence="high")
        for pt, (key, score) in ranked[:top_k]
    ]


def solve_vigenere(ciphertext: str, top_k: int = 3) -> list[SolveResult]:
    def column_decrypt(column: str, shift: int) -> str:
        return "".join(ALPHABET[(ALPHABET.index(c) - shift) % 26] for c in column)

    return _solve_family(
        ciphertext, decrypt_vigenere, column_decrypt,
        shift_range=range(26), key_symbol_fn=lambda s: ALPHABET[s],
        alphabet_symbols=ALPHABET, top_k=top_k,
    )


def solve_beaufort(ciphertext: str, top_k: int = 3) -> list[SolveResult]:
    # Beaufort's decrypt is p = (k - c) mod 26 -- the key index is
    # subtracted *from*, not subtracted, so the column transform is
    # shift-minus-char rather than char-minus-shift.
    def column_decrypt(column: str, shift: int) -> str:
        return "".join(ALPHABET[(shift - ALPHABET.index(c)) % 26] for c in column)

    return _solve_family(
        ciphertext, decrypt_beaufort, column_decrypt,
        shift_range=range(26), key_symbol_fn=lambda s: ALPHABET[s],
        alphabet_symbols=ALPHABET, top_k=top_k,
    )


def solve_gronsfeld(ciphertext: str, top_k: int = 3) -> list[SolveResult]:
    def column_decrypt(column: str, shift: int) -> str:
        return "".join(ALPHABET[(ALPHABET.index(c) - shift) % 26] for c in column)

    return _solve_family(
        ciphertext, decrypt_gronsfeld, column_decrypt,
        shift_range=range(10), key_symbol_fn=lambda s: str(s),
        alphabet_symbols="0123456789", top_k=top_k,
    )
