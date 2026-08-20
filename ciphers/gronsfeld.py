"""
Gronsfeld Cipher.

A Vigenere variant that uses a short sequence of DIGITS (0-9) as the key
instead of a keyword of letters. Historically popular because a numeric
key (like a date or a phone number) was easier for people to memorize and
carry around than an arbitrary keyword -- but it also means the key space
is only 10^L instead of Vigenere's 26^L for the same key length, which is
exactly why it was easier to break in practice.
"""
from utils.text_utils import ALPHABET


def _validate_digit_key(key: str) -> str:
    if not key or not key.isdigit():
        raise ValueError("Gronsfeld key must be a non-empty string of digits (0-9).")
    return key


def encrypt_gronsfeld(text: str, key: str) -> str:
    """Encrypts text using the Gronsfeld cipher with a numeric key."""
    key = _validate_digit_key(key)
    text = text.upper()

    result = ""
    key_i = 0
    for char in text:
        if char in ALPHABET:
            shift = int(key[key_i % len(key)])
            idx = (ALPHABET.index(char) + shift) % 26
            result += ALPHABET[idx]
            key_i += 1
        else:
            result += char
    return result


def decrypt_gronsfeld(text: str, key: str) -> str:
    """Decrypts text using the Gronsfeld cipher with a numeric key."""
    key = _validate_digit_key(key)
    text = text.upper()

    result = ""
    key_i = 0
    for char in text:
        if char in ALPHABET:
            shift = int(key[key_i % len(key)])
            idx = (ALPHABET.index(char) - shift) % 26
            result += ALPHABET[idx]
            key_i += 1
        else:
            result += char
    return result
