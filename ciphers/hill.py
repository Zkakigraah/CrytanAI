import math
import string
import sympy as sp

ALPHABET = string.ascii_uppercase

def _text_to_numbers(text: str) -> list[int]:
    """Converts a string of letters into a list of numbers (A=0 ... Z=25)."""
    return [ord(c) - 65 for c in text if c in ALPHABET]

def _numbers_to_text(numbers: list[int]) -> str:
    """Converts a list of numbers back into a string of letters."""
    return "".join([chr(n + 65) for n in numbers])

def _get_key_matrix(key: str) -> sp.Matrix:
    """Validates the key and converts it into a SymPy Matrix."""
    key = "".join([c for c in key.upper() if c in ALPHABET])
    klen = len(key)
    n = int(math.isqrt(klen))
    
    if n * n != klen:
        raise ValueError("Key length (excluding non-letters) must be a perfect square (e.g., 4, 9, 16)")
    
    nums = _text_to_numbers(key)
    # Reshape the list into an NxN matrix
    matrix_data = [nums[i * n : (i + 1) * n] for i in range(n)]
    return sp.Matrix(matrix_data)

def encrypt_hill(message: str, key: str) -> str:
    """
    Encrypts a message using a Hill cipher with the given key.
    Pads the message with 'X' if it doesn't divide evenly by the matrix size.
    Preserves non-alphabetic characters in their original positions.
    """
    message = message.upper()
    key_matrix = _get_key_matrix(key)
    n = key_matrix.shape[0]

    # Extract only letters, keep track of original positions
    letters = [(i, c) for i, c in enumerate(message) if c in ALPHABET]
    
    if not letters:
        return message

    # Pad letters list so its length is a multiple of n
    while len(letters) % n != 0:
        letters.append((len(message), "X"))
        message += "X"

    ciphertext_chars = list(message)
    block = []
    
    for idx, c in letters:
        block.append(ord(c) - 65)
        if len(block) == n:
            # Multiply matrix by block vector, mod 26
            vector = sp.Matrix(block)
            enc_vector = (key_matrix * vector) % 26
            
            # Place encrypted characters back into their original positions
            for (pos, _), val in zip(letters[:n], enc_vector):
                ciphertext_chars[pos] = chr(val + 65)
                
            letters = letters[n:]
            block = []

    return "".join(ciphertext_chars)

def decrypt_hill(message: str, key: str) -> str:
    """
    Decrypts a Hill cipher message. Requires the key matrix to be invertible modulo 26.
    """
    message = message.upper()
    key_matrix = _get_key_matrix(key)
    n = key_matrix.shape[0]

    # To decrypt, we need the modular inverse of the matrix modulo 26.
    try:
        inv_matrix = key_matrix.inv_mod(26)
    except ValueError:
        raise ValueError("This key matrix is not invertible modulo 26. It cannot be used for decryption.")

    # Extract only letters
    letters = [(i, c) for i, c in enumerate(message) if c in ALPHABET]
    
    if not letters:
        return message

    ciphertext_chars = list(message)
    block = []
    
    for idx, c in letters:
        block.append(ord(c) - 65)
        if len(block) == n:
            # Multiply inverted matrix by block vector, mod 26
            vector = sp.Matrix(block)
            dec_vector = (inv_matrix * vector) % 26
            
            for (pos, _), val in zip(letters[:n], dec_vector):
                ciphertext_chars[pos] = chr(val + 65)
                
            letters = letters[n:]
            block = []

    return "".join(ciphertext_chars)