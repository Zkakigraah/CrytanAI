"""
Keyword Cipher.

A monoalphabetic substitution cipher whose 26-letter cipher alphabet isn't
a random shuffle (see substitution.py) but is *derived* from a memorable
keyword: write the keyword's unique letters first, then the rest of the
alphabet in order. This was the practical, pre-computer way people
actually generated substitution keys they could remember instead of
writing down.

Mathematically this is a special case of the general monoalphabetic
substitution cipher -- ciphertext-only cryptanalysis can't tell a keyword
cipher apart from a randomly-shuffled substitution cipher, so
solvers/substitution_solver.py handles both identically.
"""
from utils.text_utils import ALPHABET, build_keyed_alphabet


def generate_keyword_key(keyword: str) -> str:
    """Builds the 26-letter cipher alphabet for a given keyword."""
    if not keyword or not keyword.isalpha():
        raise ValueError("Keyword must be a non-empty alphabetic string.")
    return build_keyed_alphabet(keyword, merge_j_into_i=False)


def encrypt_keyword(text: str, keyword: str) -> str:
    """Encrypts text using a keyword-derived substitution alphabet."""
    key = generate_keyword_key(keyword)
    mapping = {ALPHABET[i]: key[i] for i in range(26)}

    result = ""
    for char in text.upper():
        result += mapping.get(char, char)
    return result


def decrypt_keyword(text: str, keyword: str) -> str:
    """Decrypts text using a keyword-derived substitution alphabet."""
    key = generate_keyword_key(keyword)
    reverse_map = {key[i]: ALPHABET[i] for i in range(26)}

    result = ""
    for char in text.upper():
        result += reverse_map.get(char, char)
    return result
