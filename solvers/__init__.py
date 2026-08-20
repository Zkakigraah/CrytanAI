from .base import SolveResult
from .caesar_solver import solve_caesar
from .affine_solver import solve_affine
from .atbash_solver import solve_atbash
from .substitution_solver import solve_substitution
from .vigenere_family_solver import solve_vigenere, solve_beaufort, solve_gronsfeld
from .transposition_solver import solve_railfence, solve_scytale, solve_columnar
from .autokey_solver import solve_autokey
from .playfair_solver import solve_playfair
from .hill_solver import decrypt_known_key, crack_known_plaintext

__all__ = [
    "SolveResult",
    "solve_caesar",
    "solve_affine",
    "solve_atbash",
    "solve_substitution",
    "solve_vigenere",
    "solve_beaufort",
    "solve_gronsfeld",
    "solve_railfence",
    "solve_scytale",
    "solve_columnar",
    "solve_autokey",
    "solve_playfair",
    "decrypt_known_key",
    "crack_known_plaintext",
]

# solve_keyword and solve_polybius are deliberately absent: keyword
# cipher ciphertext is statistically identical to general substitution
# (solve_substitution covers both -- see substitution_solver.py's
# docstring), and Polybius's coordinate-pair ciphertext isn't a letter
# stream at all, so cracking an unknown *keyed* grid would need its own
# dedicated solver -- not built here since Polybius is primarily a
# format-detection case (all-digit ciphertext) rather than a
# frequency-analysis one; the unkeyed standard grid needs no solving.
