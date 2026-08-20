"""
Beaufort Cipher.

A polyalphabetic cipher closely related to Vigenere but with the shift
direction reversed: C = (K - P) mod 26 instead of Vigenere's
C = (P + K) mod 26. That single sign flip gives Beaufort an unusual,
useful property: encryption and decryption are the *identical* operation.
Applying the same transform twice with the same key returns the original
text -- there's no separate decrypt algorithm, only the same transform
run again (this module still exposes both function names for a
consistent interface with every other cipher; decrypt just calls the
same core transform as encrypt).
"""
from utils.text_utils import ALPHABET


def _beaufort_shift(char: str, key_char: str) -> str:
    if char not in ALPHABET:
        return char
    p_idx = ALPHABET.index(char)
    k_idx = ALPHABET.index(key_char)
    return ALPHABET[(k_idx - p_idx) % 26]


def _beaufort_transform(text: str, key: str) -> str:
    if not key.isalpha():
        raise ValueError("Beaufort key must only contain alphabetic characters.")

    text = text.upper()
    key = key.upper()

    result = ""
    key_i = 0
    for char in text:
        if char in ALPHABET:
            result += _beaufort_shift(char, key[key_i % len(key)])
            key_i += 1
        else:
            result += char
    return result


def encrypt_beaufort(text: str, key: str) -> str:
    """Encrypts text using the Beaufort cipher."""
    return _beaufort_transform(text, key)


def decrypt_beaufort(text: str, key: str) -> str:
    """Decrypts text using the Beaufort cipher (the identical operation
    as encryption -- see module docstring)."""
    return _beaufort_transform(text, key)
