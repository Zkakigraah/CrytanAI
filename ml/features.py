"""
Feature extraction for the cipher-family/type classifier.

Turns raw ciphertext into a fixed-length numeric feature vector. Every
feature here is something a human cryptanalyst would actually look at --
this isn't a black box: index of coincidence and chi-squared are the two
classical tools for telling transposition, monoalphabetic, and
polyalphabetic ciphers apart (verified directly against real ciphertext
from every cipher in this project -- see this project's dev notes), and
the rest (digit ratio, token-length statistics, IC-scan peakiness) target
specific, explainable signatures: Polybius's all-digit output, Playfair's
regular 2-character digraph groups, and the presence or absence of a
short repeating key period.

This is honestly a *harder* classification problem than it might look:
several ciphers are genuinely statistically indistinguishable from each
other ciphertext-only (Caesar/Affine/Substitution/Keyword all relabel
letters with no other observable trace of which specific relabeling rule
was used; Vigenere/Beaufort/Gronsfeld all cycle a short key the same
way). That's not a weakness of these features -- it's a real
information-theoretic limit on what's recoverable without attempting to
decrypt. ml/cipher_detector.py's docstring covers how the rest of the
pipeline works around that.
"""
import string
from collections import Counter

from utils.text_utils import ALPHABET, clean_letters_only, index_of_coincidence
from utils.language_stats import (
    chi_squared_statistic,
    bigram_log_likelihood,
    quadgram_log_likelihood,
    dictionary_word_ratio,
)

FEATURE_NAMES = [
    "index_of_coincidence",
    "chi_squared",
    "bigram_log_likelihood",
    "quadgram_log_likelihood",
    "word_ratio",
    "digit_ratio",
    "space_ratio",
    "avg_token_length",
    "token_length_variance",
    "ic_scan_peakiness",
    "length",
]

CHI2_CAP = 5000.0  # sklearn can't handle inf (Polybius's near-empty-letter
                    # case produces it) -- cap rather than drop the signal.


def _ic_scan_peakiness(letters: str, max_length: int = 12) -> float:
    """How much higher the best candidate key-length's average column IC
    is than the text's own overall IC. High for a cipher with a short
    repeating key (Vigenere/Beaufort/Gronsfeld -- some length reveals
    clean single-alphabet columns); low/flat for ciphers with no such
    period (monoalphabetic, transposition, and notably Autokey, whose
    effective key is nearly the message length)."""
    if len(letters) < 20:
        return 0.0

    base_ic = index_of_coincidence(letters)
    best = base_ic
    for length in range(2, max_length + 1):
        columns = [letters[i::length] for i in range(length)]
        ics = [index_of_coincidence(c) for c in columns if len(c) >= 2]
        if ics:
            best = max(best, sum(ics) / len(ics))
    return best - base_ic


def extract_features(ciphertext: str) -> list[float]:
    """Returns a feature vector in the fixed order given by FEATURE_NAMES."""
    n = len(ciphertext)
    letters = clean_letters_only(ciphertext)
    digits = sum(1 for ch in ciphertext if ch.isdigit())
    spaces = sum(1 for ch in ciphertext if ch == " ")
    tokens = ciphertext.split()
    token_lengths = [len(t) for t in tokens] or [0]
    avg_token_len = sum(token_lengths) / len(token_lengths)
    token_len_var = sum((t - avg_token_len) ** 2 for t in token_lengths) / len(token_lengths)

    chi2 = chi_squared_statistic(ciphertext)
    if chi2 == float("inf"):
        chi2 = CHI2_CAP
    chi2 = min(chi2, CHI2_CAP)

    return [
        index_of_coincidence(ciphertext),
        chi2,
        bigram_log_likelihood(ciphertext),
        quadgram_log_likelihood(ciphertext),
        dictionary_word_ratio(ciphertext),
        digits / n if n else 0.0,
        spaces / n if n else 0.0,
        avg_token_len,
        token_len_var,
        _ic_scan_peakiness(letters),
        float(n),
    ]
