import string

ALPHABET = string.ascii_uppercase

def encrypt_caesar(text: str, shift: int) -> str:
    """Encrypts text using the Caesar cipher with a given shift."""
    text = text.upper()
    result = ""

    for char in text:
        if char in ALPHABET:
            idx = ALPHABET.index(char)
            new_idx = (idx + shift) % 26
            result += ALPHABET[new_idx]
        else:
            # Preserve punctuation and spaces
            result += char

    return result

def decrypt_caesar(text: str, shift: int) -> str:
    """Decrypts text using the Caesar cipher with a given shift."""
    # Decrypting is simply encrypting with the negative shift
    return encrypt_caesar(text, -shift)