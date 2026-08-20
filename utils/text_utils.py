"""
Shared text-handling utilities.

Nothing in this file knows what a cipher is. It's pure string plumbing --
cleaning, alphabet-only extraction, and keyed-alphabet construction -- that
several *independent* cipher modules and solvers happen to need. Keeping it
here (instead of importing one cipher file from another) is what lets every
module in ciphers/ stay self-contained: they all depend on utils/, never on
each other.
"""
import string

ALPHABET = string.ascii_uppercase


def clean_letters_only(text: str) -> str:
    """Uppercase, alphabetic characters only. Used by ciphers whose
    mathematical structure (e.g. digraph substitution, block matrices)
    doesn't have a sensible way to preserve punctuation/spacing inline."""
    return "".join(ch for ch in text.upper() if ch in ALPHABET)


def only_alpha_positions(text: str) -> list[tuple[int, str]]:
    """Returns [(index, letter), ...] for every alphabetic character in
    text, preserving its original position. Useful for ciphers that
    process letters in blocks but must place results back into a
    punctuation-preserving output string."""
    upper = text.upper()
    return [(i, ch) for i, ch in enumerate(upper) if ch in ALPHABET]


def index_of_coincidence(text: str) -> float:
    """
    The Index of Coincidence: the probability that two randomly chosen
    letters from the text are the same. Computed directly from the text's
    own letter distribution -- no reference table needed.

    Typical values:
      ~0.066-0.070  monoalphabetic substitution / natural English
      ~0.038-0.050  polyalphabetic (Vigenere-family) ciphertext
      ~0.0385       uniformly random letters

    This is the single most useful "which cipher family is this?" signal
    for classical ciphers, and it costs nothing to compute -- it's why
    it's implemented here as a pure text property rather than inside the
    ml/ package.
    """
    letters = clean_letters_only(text)
    n = len(letters)
    if n < 2:
        return 0.0

    counts = {}
    for ch in letters:
        counts[ch] = counts.get(ch, 0) + 1

    numerator = sum(c * (c - 1) for c in counts.values())
    denominator = n * (n - 1)
    return numerator / denominator


def build_keyed_alphabet(keyword: str, merge_j_into_i: bool = True) -> str:
    """
    Builds a 25- or 26-letter keyed alphabet: the keyword's unique letters
    first (in order of first appearance), followed by the remaining
    alphabet letters in order.

    Shared by ciphers/keyword.py, ciphers/polybius.py, and
    ciphers/playfair.py -- all three build a keyed alphabet from a keyword,
    just with a different number of cells (26 vs 25-with-I/J-merged) and a
    different encoding rule on top. That's exactly the kind of shared
    *utility* (not cipher logic) this module exists for: each of those
    three ciphers still encrypts/decrypts completely independently.

    Args:
        keyword: the keyword to seed the alphabet with. Non-letters are
            ignored; repeated letters are collapsed to their first
            occurrence.
        merge_j_into_i: if True, J is dropped from the base alphabet
            (classic Polybius/Playfair convention: I and J share a cell),
            producing a 25-letter result. If False, produces the full
            26-letter result (used by the plain keyword cipher).
    """
    base = ALPHABET.replace("J", "") if merge_j_into_i else ALPHABET

    seen = []
    seen_set = set()
    for ch in keyword.upper():
        if ch == "J" and merge_j_into_i:
            ch = "I"
        if ch in base and ch not in seen_set:
            seen.append(ch)
            seen_set.add(ch)

    remaining = [ch for ch in base if ch not in seen_set]
    return "".join(seen) + "".join(remaining)


def chunk(seq, size: int):
    """Yields successive `size`-length chunks of seq."""
    for i in range(0, len(seq), size):
        yield seq[i:i + size]
