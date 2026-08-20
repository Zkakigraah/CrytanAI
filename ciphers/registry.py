"""
Central cipher registry.

Every other file in ciphers/ is completely self-contained -- none of them
import each other, and none of them know this file exists. This is the
deliberate exception: the one place in the project allowed to know about
every cipher, so that solvers/, ml/, and app/ don't each need their own
hardcoded list of "which ciphers exist and how do I call them". Add a new
cipher by writing its module (independent, like all the others) and
adding one CipherSpec entry here; nothing else needs to change.

Every spec exposes encrypt/decrypt through the SAME uniform interface --
encrypt(text, key) / decrypt(text, key) -- regardless of how many
arguments or what types the underlying cipher function actually takes.
Most ciphers already take exactly one (text, key) pair, so their spec
just references the original function directly; Affine (two numbers) and
Atbash (no key at all) get a two-line adapter to fit the same shape. That
uniformity is what lets the dataset generator and the Streamlit UI treat
all 15 ciphers identically.
"""
import math
import random
import string
from dataclasses import dataclass
from typing import Any, Callable

from ciphers import caesar, substitution, vigenere, hill, affine, atbash, railfence
from ciphers import keyword, polybius, beaufort, autokey, gronsfeld, playfair, columnar, scytale

ALPHABET = string.ascii_uppercase

MONOALPHABETIC = "monoalphabetic"
POLYALPHABETIC = "polyalphabetic"
POLYGRAPHIC = "polygraphic"
TRANSPOSITION = "transposition"


@dataclass
class CipherSpec:
    slug: str                              # stable id; used as the ML classifier's label
    display_name: str
    family: str
    era: str                               # short historical note, for the UI
    description: str
    encrypt: Callable[[str, Any], str]     # uniform (text, key) -> ciphertext
    decrypt: Callable[[str, Any], str]     # uniform (text, key) -> plaintext
    random_key: Callable[[], Any]          # a valid random key, same shape encrypt/decrypt expect
    key_description: str                   # human-readable, for the UI
    key_example: Any                       # one concrete example key, for the UI placeholder


# --------------------------------------------------------------------------
# Adapters for the two ciphers whose native signature isn't already a
# single (text, key) pair.
# --------------------------------------------------------------------------

def _affine_encrypt(text, key):
    a, b = key
    return affine.encrypt_affine(text, a, b)


def _affine_decrypt(text, key):
    a, b = key
    return affine.decrypt_affine(text, a, b)


def _atbash_encrypt(text, key=None):
    return atbash.encrypt_atbash(text)


def _atbash_decrypt(text, key=None):
    return atbash.decrypt_atbash(text)


# --------------------------------------------------------------------------
# Random-key generators (used by the synthetic dataset generator in
# ml/dataset_generator.py, and available to the UI for a "random key" button).
# --------------------------------------------------------------------------

def _random_shift():
    return random.randint(1, 25)


def _random_affine_key():
    valid_a = [a for a in range(1, 26) if math.gcd(a, 26) == 1]
    return (random.choice(valid_a), random.randint(0, 25))


def _random_word(min_len=5, max_len=9):
    return "".join(random.choice(ALPHABET) for _ in range(random.randint(min_len, max_len)))


def _random_substitution_key():
    letters = list(ALPHABET)
    random.shuffle(letters)
    return "".join(letters)


def _random_digit_key(min_len=3, max_len=6):
    return "".join(random.choice(string.digits) for _ in range(random.randint(min_len, max_len)))


def _random_rails():
    return random.randint(2, 6)


def _random_scytale_rows():
    return random.randint(2, 5)


def _random_hill_key():
    """A random invertible-mod-26 2x2 matrix, as a 4-letter key string.
    Rejection sampling: 2x2 determinants are cheap to check directly, so
    this doesn't need sympy or the heavier n x n machinery in hill.py."""
    while True:
        nums = [random.randint(0, 25) for _ in range(4)]
        det = (nums[0] * nums[3] - nums[1] * nums[2]) % 26
        if math.gcd(det, 26) == 1:
            return "".join(ALPHABET[x] for x in nums)


# --------------------------------------------------------------------------
# The registry itself.
# --------------------------------------------------------------------------

CIPHER_REGISTRY: dict[str, CipherSpec] = {
    "caesar": CipherSpec(
        slug="caesar", display_name="Caesar Cipher", family=MONOALPHABETIC,
        era="~50 BCE, Roman Empire",
        description="Every letter shifted by a fixed amount. The oldest cipher in this project.",
        encrypt=caesar.encrypt_caesar, decrypt=caesar.decrypt_caesar,
        random_key=_random_shift, key_description="Integer shift, 1-25", key_example=3,
    ),
    "atbash": CipherSpec(
        slug="atbash", display_name="Atbash Cipher", family=MONOALPHABETIC,
        era="~500 BCE, Hebrew scribes",
        description="Reverses the alphabet (A<->Z, B<->Y, ...). No key -- it's its own inverse.",
        encrypt=_atbash_encrypt, decrypt=_atbash_decrypt,
        random_key=lambda: None, key_description="None (fixed mapping)", key_example=None,
    ),
    "affine": CipherSpec(
        slug="affine", display_name="Affine Cipher", family=MONOALPHABETIC,
        era="Classical antiquity (generalized Caesar)",
        description="E(x) = (a*x + b) mod 26. Caesar is the special case a=1.",
        encrypt=_affine_encrypt, decrypt=_affine_decrypt,
        random_key=_random_affine_key, key_description="(a, b) with gcd(a,26)=1", key_example=(5, 8),
    ),
    "substitution": CipherSpec(
        slug="substitution", display_name="Substitution Cipher", family=MONOALPHABETIC,
        era="9th century onward (formalized by Al-Kindi's frequency analysis)",
        description="Each letter mapped to an arbitrary, unique replacement letter.",
        encrypt=substitution.encrypt_substitution, decrypt=substitution.decrypt_substitution,
        random_key=_random_substitution_key, key_description="26-letter permutation of the alphabet",
        key_example="ZEBRACDFGHIJKLMNOPQSTUVWXY",
    ),
    "keyword": CipherSpec(
        slug="keyword", display_name="Keyword Cipher", family=MONOALPHABETIC,
        era="Renaissance-era practical cryptography",
        description="A substitution alphabet built from a memorable keyword instead of a random shuffle.",
        encrypt=keyword.encrypt_keyword, decrypt=keyword.decrypt_keyword,
        random_key=lambda: _random_word(5, 8), key_description="A keyword (letters only)", key_example="ZEBRA",
    ),
    "polybius": CipherSpec(
        slug="polybius", display_name="Polybius Square", family=MONOALPHABETIC,
        era="~150 BCE, Ancient Greece",
        description="Each letter becomes a 2-digit grid coordinate. I and J share a cell.",
        encrypt=polybius.encrypt_polybius, decrypt=polybius.decrypt_polybius,
        random_key=lambda: _random_word(4, 7), key_description="Optional keyword (blank = standard grid)",
        key_example="",
    ),
    "vigenere": CipherSpec(
        slug="vigenere", display_name="Vigenere Cipher", family=POLYALPHABETIC,
        era="16th century (often misattributed; Bellaso, 1553)",
        description="Caesar shifts that cycle through a repeating keyword, one letter of key per letter of text.",
        encrypt=vigenere.encrypt_vigenere, decrypt=vigenere.decrypt_vigenere,
        random_key=lambda: _random_word(5, 9), key_description="A keyword (letters only)", key_example="LEMON",
    ),
    "beaufort": CipherSpec(
        slug="beaufort", display_name="Beaufort Cipher", family=POLYALPHABETIC,
        era="18th century (Francis Beaufort)",
        description="Like Vigenere with the shift reversed (C = K - P). Self-reciprocal: encrypt = decrypt.",
        encrypt=beaufort.encrypt_beaufort, decrypt=beaufort.decrypt_beaufort,
        random_key=lambda: _random_word(5, 9), key_description="A keyword (letters only)", key_example="FORTIFY",
    ),
    "autokey": CipherSpec(
        slug="autokey", display_name="Autokey Cipher", family=POLYALPHABETIC,
        era="16th century (Bellaso / Vigenere)",
        description="Vigenere with a short primer key, then the keystream continues as the plaintext itself.",
        encrypt=autokey.encrypt_autokey, decrypt=autokey.decrypt_autokey,
        random_key=lambda: _random_word(3, 6), key_description="A short primer keyword", key_example="KEY",
    ),
    "gronsfeld": CipherSpec(
        slug="gronsfeld", display_name="Gronsfeld Cipher", family=POLYALPHABETIC,
        era="17th century (Count of Gronsfeld)",
        description="Vigenere with a digit-string key (0-9) instead of a keyword -- easier to memorize, weaker.",
        encrypt=gronsfeld.encrypt_gronsfeld, decrypt=gronsfeld.decrypt_gronsfeld,
        random_key=lambda: _random_digit_key(3, 6), key_description="A string of digits", key_example="3141",
    ),
    "hill": CipherSpec(
        slug="hill", display_name="Hill Cipher", family=POLYGRAPHIC,
        era="1929 (Lester S. Hill)",
        description="Encrypts blocks of letters at once via matrix multiplication mod 26.",
        encrypt=hill.encrypt_hill, decrypt=hill.decrypt_hill,
        random_key=_random_hill_key, key_description="N^2-letter key (perfect-square length; flattened matrix)",
        key_example="GYBNQKURP",
    ),
    "playfair": CipherSpec(
        slug="playfair", display_name="Playfair Cipher", family=POLYGRAPHIC,
        era="1854 (Charles Wheatstone / Lord Playfair)",
        description="Encrypts pairs of letters at once using a 5x5 keyed grid and row/column/rectangle rules.",
        encrypt=playfair.encrypt_playfair, decrypt=playfair.decrypt_playfair,
        random_key=lambda: _random_word(5, 9), key_description="Optional keyword (blank = standard grid)",
        key_example="PLAYFAIREXAMPLE",
    ),
    "railfence": CipherSpec(
        slug="railfence", display_name="Rail Fence Cipher", family=TRANSPOSITION,
        era="Classical antiquity (military field cipher)",
        description="Writes text in a zig-zag across N rails, then reads the rails off in order.",
        encrypt=railfence.encrypt_railfence, decrypt=railfence.decrypt_railfence,
        random_key=_random_rails, key_description="Integer number of rails, >=2", key_example=3,
    ),
    "columnar": CipherSpec(
        slug="columnar", display_name="Columnar Transposition", family=TRANSPOSITION,
        era="Widely used, WWI trench ciphers",
        description="Writes text into rows under a keyword, reads columns out in alphabetical key order.",
        encrypt=columnar.encrypt_columnar, decrypt=columnar.decrypt_columnar,
        random_key=lambda: _random_word(4, 7), key_description="A keyword (letters only)", key_example="ZEBRA",
    ),
    "scytale": CipherSpec(
        slug="scytale", display_name="Scytale Cipher", family=TRANSPOSITION,
        era="~7th century BCE, Ancient Sparta",
        description="Text wrapped around a rod of a given circumference; unwrapped, letters are scrambled.",
        encrypt=scytale.encrypt_scytale, decrypt=scytale.decrypt_scytale,
        random_key=_random_scytale_rows, key_description="Integer number of rows (rod circumference)",
        key_example=4,
    ),
}


def get_cipher(slug: str) -> CipherSpec:
    if slug not in CIPHER_REGISTRY:
        raise KeyError(f"Unknown cipher slug: {slug!r}. Known: {sorted(CIPHER_REGISTRY)}")
    return CIPHER_REGISTRY[slug]


def list_ciphers() -> list[CipherSpec]:
    return list(CIPHER_REGISTRY.values())


def list_slugs() -> list[str]:
    return list(CIPHER_REGISTRY.keys())


def list_families() -> dict[str, list[CipherSpec]]:
    """Groups specs by family, in a fixed display order -- handy for a
    grouped dropdown in the UI."""
    families: dict[str, list[CipherSpec]] = {
        MONOALPHABETIC: [], POLYALPHABETIC: [], POLYGRAPHIC: [], TRANSPOSITION: [],
    }
    for spec in CIPHER_REGISTRY.values():
        families[spec.family].append(spec)
    return families


def random_cipher_and_key() -> tuple[CipherSpec, Any]:
    """Picks a uniformly random cipher and generates a valid random key
    for it. Used by ml/dataset_generator.py to build labeled synthetic
    training examples."""
    spec = random.choice(list(CIPHER_REGISTRY.values()))
    return spec, spec.random_key()
