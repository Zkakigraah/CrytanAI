"""
Correctness tests for solvers/. These use longer, more realistic
passages than tests/test_ciphers.py's round-trip checks -- frequency
analysis and hill-climbing need enough text to have real statistical
signal (this was a genuine, tested finding during development: the
substitution and Vigenere-family solvers both need several hundred
characters to reliably converge, not just a few words -- see those
modules' docstrings).

Best-effort solvers (Autokey, Playfair, large-keyword Columnar) are
intentionally NOT asserted to succeed every time here -- that would test
a guarantee those solvers explicitly don't make. What's tested instead is
that they run without error and report the honest confidence tier they're
documented to report.
"""
from ciphers import encrypt_caesar, encrypt_affine, encrypt_atbash, encrypt_substitution
from ciphers import encrypt_vigenere, encrypt_beaufort, encrypt_gronsfeld
from ciphers import encrypt_railfence, encrypt_columnar, encrypt_scytale
from ciphers.substitution import generate_substitution_key
from ciphers.hill import encrypt_hill

from solvers import (
    solve_caesar, solve_affine, solve_atbash, solve_substitution,
    solve_vigenere, solve_beaufort, solve_gronsfeld,
    solve_railfence, solve_scytale, solve_columnar,
    solve_autokey, solve_playfair, decrypt_known_key, crack_known_plaintext,
)

LONG_TEXT = (
    "MACHINE LEARNING HAS TRANSFORMED THE FIELD OF CRYPTANALYSIS BY ALLOWING "
    "COMPUTERS TO AUTOMATICALLY DETECT PATTERNS IN ENCRYPTED TEXT THAT WOULD "
    "TAKE A HUMAN ANALYST MANY HOURS TO FIND BY HAND. CLASSICAL CIPHERS SUCH "
    "AS THE SUBSTITUTION CIPHER LEAVE STATISTICAL FOOTPRINTS BEHIND BECAUSE "
    "THEY PRESERVE THE UNDERLYING FREQUENCY DISTRIBUTION OF THE ORIGINAL "
    "LANGUAGE EVEN THOUGH EVERY INDIVIDUAL LETTER HAS BEEN RELABELED."
)


def test_solve_caesar_exact():
    ciphertext = encrypt_caesar(LONG_TEXT, 11)
    result = solve_caesar(ciphertext, top_k=1)[0]
    assert result.plaintext == LONG_TEXT.upper()
    assert result.confidence == "high"


def test_solve_affine_exact():
    ciphertext = encrypt_affine(LONG_TEXT, 7, 3)
    result = solve_affine(ciphertext, top_k=1)[0]
    assert result.plaintext == LONG_TEXT.upper()


def test_solve_atbash_exact():
    ciphertext = encrypt_atbash(LONG_TEXT)
    result = solve_atbash(ciphertext)[0]
    assert result.plaintext == LONG_TEXT.upper()


def test_solve_substitution_exact_on_long_text():
    key = generate_substitution_key()
    ciphertext = encrypt_substitution(LONG_TEXT, key)
    result = solve_substitution(ciphertext, top_k=1)[0]
    assert result.plaintext == LONG_TEXT.upper()


def test_solve_vigenere_family_exact():
    for encrypt_fn, solve_fn, key in [
        (encrypt_vigenere, solve_vigenere, "SECRET"),
        (encrypt_beaufort, solve_beaufort, "FORTIFY"),
        (encrypt_gronsfeld, solve_gronsfeld, "31415"),
    ]:
        ciphertext = encrypt_fn(LONG_TEXT, key)
        result = solve_fn(ciphertext, top_k=1)[0]
        assert result.plaintext == LONG_TEXT.upper()
        assert result.key == key  # exact minimal key, not a repeated-block artifact


def test_solve_transposition_exact():
    ciphertext = encrypt_railfence(LONG_TEXT, 4)
    assert solve_railfence(ciphertext, top_k=1)[0].plaintext == LONG_TEXT

    ciphertext = encrypt_scytale(LONG_TEXT, 5)
    assert solve_scytale(ciphertext, top_k=1)[0].plaintext == LONG_TEXT

    ciphertext = encrypt_columnar(LONG_TEXT, "ZEBRA")  # 5 columns: exhaustive, reliable regime
    assert solve_columnar(ciphertext, top_k=1)[0].plaintext == LONG_TEXT


def test_solve_autokey_runs_and_reports_best_effort():
    ciphertext = encrypt_vigenere(LONG_TEXT, "KEY")  # any ciphertext is fine -- checking it runs cleanly
    results = solve_autokey(ciphertext, top_k=3)
    assert len(results) >= 1
    assert all(r.confidence == "best_effort" for r in results)


def test_solve_playfair_runs_and_reports_best_effort():
    from ciphers.playfair import encrypt_playfair

    ciphertext = encrypt_playfair(LONG_TEXT, "EXAMPLE")
    results = solve_playfair(ciphertext, top_k=3)
    assert len(results) >= 1
    assert all(r.confidence == "best_effort" for r in results)


def test_hill_known_key_passthrough():
    key = "GYBN"  # NOTE: not invertible mod 26 -- deliberately testing that
    # encrypt still works (hill.py doesn't validate at encrypt time) while
    # decrypt correctly refuses, rather than silently producing garbage.
    ciphertext = encrypt_hill(LONG_TEXT, key)
    import pytest as _pytest
    with _pytest.raises(ValueError):
        decrypt_known_key(ciphertext, key)


def test_hill_known_key_passthrough_valid_key():
    from sympy import Matrix
    from utils.text_utils import ALPHABET
    import random

    random.seed(0)
    while True:
        nums = [random.randint(0, 25) for _ in range(4)]
        try:
            Matrix(2, 2, nums).inv_mod(26)
            break
        except Exception:
            continue
    key = "".join(ALPHABET[n] for n in nums)
    ciphertext = encrypt_hill(LONG_TEXT, key)
    result = decrypt_known_key(ciphertext, key)
    assert result.plaintext.startswith("MACHINE")


def test_hill_crib_attack_exact_key_recovery():
    from sympy import Matrix
    from utils.text_utils import ALPHABET, clean_letters_only
    import random

    random.seed(1)
    n = 2
    while True:
        nums = [random.randint(0, 25) for _ in range(n * n)]
        try:
            Matrix(n, n, nums).inv_mod(26)
            break
        except Exception:
            continue
    true_key = "".join(ALPHABET[x] for x in nums)
    ciphertext = encrypt_hill(LONG_TEXT, true_key)

    # Not every crib's plaintext matrix is invertible mod 26 -- that's a
    # real, tested, documented constraint of the attack (see
    # crack_known_plaintext's docstring), not something to work around by
    # assuming any hardcoded slice of text happens to qualify. Search for
    # a block-aligned offset that does, the same way a real cryptanalyst
    # would if their first guess at crib placement didn't pan out.
    letters = clean_letters_only(LONG_TEXT)
    offset = None
    for candidate_offset in range(0, len(letters) - n * n, n):
        crib = letters[candidate_offset:candidate_offset + n * n]
        crib_nums = [ALPHABET.index(c) for c in crib]
        try:
            Matrix(n, n, lambda r, c: crib_nums[c * n + r]).inv_mod(26)
            offset = candidate_offset
            break
        except Exception:
            continue
    assert offset is not None, "no invertible crib found in LONG_TEXT -- test text needs adjusting"

    crib = letters[offset:offset + n * n]
    result = crack_known_plaintext(ciphertext, crib, block_size=n, ciphertext_offset=offset)
    assert result.key == true_key


def test_hill_crib_attack_rejects_misaligned_offset():
    import pytest as _pytest

    ciphertext = encrypt_hill(LONG_TEXT, "ABCD")
    with _pytest.raises(ValueError, match="multiple of block_size"):
        crack_known_plaintext(ciphertext, "MACHINELEA", block_size=3, ciphertext_offset=1)


def test_hill_crib_attack_rejects_short_crib():
    import pytest as _pytest

    ciphertext = encrypt_hill(LONG_TEXT, "ABCD")
    with _pytest.raises(ValueError, match="at least"):
        crack_known_plaintext(ciphertext, "AB", block_size=3)
