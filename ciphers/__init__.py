from .caesar import encrypt_caesar, decrypt_caesar
from .substitution import generate_substitution_key, encrypt_substitution, decrypt_substitution
from .vigenere import encrypt_vigenere, decrypt_vigenere
from .hill import encrypt_hill, decrypt_hill
from .affine import encrypt_affine, decrypt_affine
from .atbash import encrypt_atbash, decrypt_atbash
from .railfence import encrypt_railfence, decrypt_railfence
from .keyword import generate_keyword_key, encrypt_keyword, decrypt_keyword
from .polybius import encrypt_polybius, decrypt_polybius
from .beaufort import encrypt_beaufort, decrypt_beaufort
from .autokey import encrypt_autokey, decrypt_autokey
from .gronsfeld import encrypt_gronsfeld, decrypt_gronsfeld
from .playfair import encrypt_playfair, decrypt_playfair
from .columnar import encrypt_columnar, decrypt_columnar
from .scytale import encrypt_scytale, decrypt_scytale

__all__ = [
    "encrypt_caesar", "decrypt_caesar",
    "generate_substitution_key", "encrypt_substitution", "decrypt_substitution",
    "encrypt_vigenere", "decrypt_vigenere",
    "encrypt_hill", "decrypt_hill",
    "encrypt_affine", "decrypt_affine",
    "encrypt_atbash", "decrypt_atbash",
    "encrypt_railfence", "decrypt_railfence",
    "generate_keyword_key", "encrypt_keyword", "decrypt_keyword",
    "encrypt_polybius", "decrypt_polybius",
    "encrypt_beaufort", "decrypt_beaufort",
    "encrypt_autokey", "decrypt_autokey",
    "encrypt_gronsfeld", "decrypt_gronsfeld",
    "encrypt_playfair", "decrypt_playfair",
    "encrypt_columnar", "decrypt_columnar",
    "encrypt_scytale", "decrypt_scytale",
]

# The registry is imported separately (from ciphers.registry import ...) by
# solvers/, ml/, and app/ rather than re-exported here. It depends on every
# module above already being importable, and keeping it out of this flat
# namespace makes the distinction clear: the names above are the 15
# independent cipher implementations; ciphers.registry is the one place
# that's allowed to know about all of them at once.
