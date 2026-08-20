import string

ALPHABET = string.ascii_uppercase
REVERSE_ALPHABET = ALPHABET[::-1]

def encrypt_atbash(text: str) -> str:
    """
    Encrypts text using the Atbash cipher (reverses the alphabet).
    Because it is symmetric, encryption and decryption are the exact same process.
    """
    mapping = {ALPHABET[i]: REVERSE_ALPHABET[i] for i in range(26)}
    
    result = ""
    for char in text.upper():
        if char in mapping:
            result += mapping[char]
        else:
            result += char
            
    return result

def decrypt_atbash(text: str) -> str:
    """
    Decrypts text using the Atbash cipher. 
    Calls the encryption function since Atbash is its own inverse.
    """
    return encrypt_atbash(text)