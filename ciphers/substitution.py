import random
import string

ALPHABET = string.ascii_uppercase

def generate_substitution_key() -> str:
    """Generates a random 26-letter key for the substitution cipher."""
    letters = list(ALPHABET)
    random.shuffle(letters)
    return "".join(letters)

def encrypt_substitution(text: str, key: str) -> str:
    """Encrypts text using a mapped 26-letter substitution key."""
    if len(key) != 26:
        raise ValueError("Key must be exactly 26 letters long.")
        
    text = text.upper()
    key = key.upper()
    
    # Create a mapping from standard alphabet to the randomized key
    mapping = {ALPHABET[i]: key[i] for i in range(26)}

    result = ""
    for char in text:
        if char in mapping:
            result += mapping[char]
        else:
            result += char
            
    return result

def decrypt_substitution(text: str, key: str) -> str:
    """Decrypts text using the mapped 26-letter substitution key."""
    text = text.upper()
    key = key.upper()
    
    # Create a reverse mapping from the randomized key back to the standard alphabet
    reverse_map = {key[i]: ALPHABET[i] for i in range(26)}

    result = ""
    for char in text:
        if char in reverse_map:
            result += reverse_map[char]
        else:
            result += char
            
    return result