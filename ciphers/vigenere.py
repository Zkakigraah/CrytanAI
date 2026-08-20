import string

ALPHABET = string.ascii_uppercase

def _shift_char(char: str, key_char: str, encrypt: bool = True) -> str:
    """Helper function to shift a single character based on a key character."""
    if char not in ALPHABET:
        return char

    t_idx = ALPHABET.index(char)
    k_idx = ALPHABET.index(key_char)

    if encrypt:
        new_idx = (t_idx + k_idx) % 26
    else:
        new_idx = (t_idx - k_idx) % 26

    return ALPHABET[new_idx]

def encrypt_vigenere(text: str, key: str) -> str:
    """Encrypts text using the Vigenère cipher and a keyword."""
    if not key.isalpha():
        raise ValueError("Vigenère key must only contain alphabetic characters.")
        
    text = text.upper()
    key = key.upper()

    result = ""
    key_i = 0

    for char in text:
        if char in ALPHABET:
            result += _shift_char(char, key[key_i % len(key)], encrypt=True)
            key_i += 1
        else:
            result += char

    return result

def decrypt_vigenere(text: str, key: str) -> str:
    """Decrypts text using the Vigenère cipher and a keyword."""
    if not key.isalpha():
        raise ValueError("Vigenère key must only contain alphabetic characters.")
        
    text = text.upper()
    key = key.upper()

    result = ""
    key_i = 0

    for char in text:
        if char in ALPHABET:
            result += _shift_char(char, key[key_i % len(key)], encrypt=False)
            key_i += 1
        else:
            result += char

    return result