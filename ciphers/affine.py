import string
import math

ALPHABET = string.ascii_uppercase

def encrypt_affine(text: str, a: int, b: int) -> str:
    """
    Encrypts using Affine cipher: E(x) = (ax + b) mod 26.
    'a' must be coprime to 26 (e.g., 1, 3, 5, 7, 9, 11, 15, 17, 19, 21, 23, 25).
    """
    if math.gcd(a, 26) != 1:
        raise ValueError(f"Key 'a' ({a}) must be coprime to 26.")
        
    result = ""
    for char in text.upper():
        if char in ALPHABET:
            idx = ALPHABET.index(char)
            new_idx = (a * idx + b) % 26
            result += ALPHABET[new_idx]
        else:
            result += char
    return result

def decrypt_affine(text: str, a: int, b: int) -> str:
    """
    Decrypts using Affine cipher: D(x) = a^-1(x - b) mod 26.
    """
    if math.gcd(a, 26) != 1:
        raise ValueError(f"Key 'a' ({a}) must be coprime to 26.")
        
    result = ""
    # pow(a, -1, 26) finds the modular multiplicative inverse of 'a' modulo 26
    a_inv = pow(a, -1, 26)
    
    for char in text.upper():
        if char in ALPHABET:
            idx = ALPHABET.index(char)
            new_idx = (a_inv * (idx - b)) % 26
            result += ALPHABET[new_idx]
        else:
            result += char
    return result