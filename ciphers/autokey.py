"""
Autokey Cipher.

A polyalphabetic cipher that fixes Vigenere's main weakness (a short,
repeating key) by using a short "primer" key for only the first few
letters, then extending the keystream with the plaintext itself:

    keystream = primer + plaintext

Because the keystream (almost) never repeats, Kasiski examination and the
Friedman test -- the standard tools for finding a Vigenere key's length --
don't find a meaningful period here; the effective key is nearly as long
as the message. This is exactly why solvers/autokey_solver.py treats this
cipher as a best-effort search over primer length and content rather than
a closed-form attack (see that module's docstring for details).

Only alphabetic characters participate in and advance the keystream,
matching the convention used by vigenere.py -- spaces and punctuation are
copied through untouched and don't consume a keystream position.
"""
from utils.text_utils import ALPHABET


def encrypt_autokey(text: str, primer: str) -> str:
    """Encrypts text using the Autokey cipher with the given primer key."""
    if not primer.isalpha():
        raise ValueError("Autokey primer must only contain alphabetic characters.")

    text = text.upper()
    primer = primer.upper()

    plaintext_letters = [ch for ch in text if ch in ALPHABET]
    keystream = list(primer) + plaintext_letters

    result = ""
    key_i = 0
    for char in text:
        if char in ALPHABET:
            p_idx = ALPHABET.index(char)
            k_idx = ALPHABET.index(keystream[key_i])
            result += ALPHABET[(p_idx + k_idx) % 26]
            key_i += 1
        else:
            result += char
    return result


def decrypt_autokey(text: str, primer: str) -> str:
    """
    Decrypts text using the Autokey cipher with the given primer key.

    Unlike every other cipher in this project, this can't be a simple
    character-by-character map: the keystream for position i (once past
    the primer) IS the plaintext letter this same loop already recovered
    earlier, at position i - len(primer). So decryption builds up the
    recovered plaintext sequentially, feeding on its own output as it
    goes.
    """
    if not primer.isalpha():
        raise ValueError("Autokey primer must only contain alphabetic characters.")

    text = text.upper()
    primer = primer.upper()
    primer_len = len(primer)

    recovered_letters = []
    result = ""

    for char in text:
        if char in ALPHABET:
            j = len(recovered_letters)
            key_char = primer[j] if j < primer_len else recovered_letters[j - primer_len]

            c_idx = ALPHABET.index(char)
            k_idx = ALPHABET.index(key_char)
            p_char = ALPHABET[(c_idx - k_idx) % 26]

            recovered_letters.append(p_char)
            result += p_char
        else:
            result += char
    return result
