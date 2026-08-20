"""
Polybius Square Cipher.

Ancient Greek cipher (attributed to the historian Polybius, ~2nd century
BCE) that maps each letter to a coordinate pair on a 5x5 grid. I and J
share a cell (the grid only has 25 slots for 26 letters) -- decryption
always recovers I, never J. Takes an optional keyword to build a keyed
grid instead of the plain A-Z grid; with no keyword, the standard unkeyed
grid is used.

Ciphertext is digits only (two digits per letter: row then column, both
1-5). Non-alphabetic characters (spaces, punctuation) pass through
unchanged between digit groups, so word boundaries stay visible.

Known limitation, inherent to an all-digit ciphertext alphabet rather than
a bug: if the original plaintext itself contains literal digit
characters, decrypt cannot distinguish them from encoded letter-pairs.
This cipher's domain is letters; mixed alphanumeric input isn't
well-defined for it.
"""
from utils.text_utils import build_keyed_alphabet

GRID_SIZE = 5


def _build_grid(keyword: str = "") -> tuple[dict, dict]:
    """Returns (letter_to_coords, coords_to_letter) for a 5x5 grid seeded
    by `keyword` (empty string = standard A-Z grid, J merged into I)."""
    alphabet_25 = build_keyed_alphabet(keyword, merge_j_into_i=True)

    letter_to_coords = {}
    coords_to_letter = {}
    for idx, ch in enumerate(alphabet_25):
        row, col = divmod(idx, GRID_SIZE)
        coord = f"{row + 1}{col + 1}"
        letter_to_coords[ch] = coord
        coords_to_letter[coord] = ch

    return letter_to_coords, coords_to_letter


def encrypt_polybius(text: str, keyword: str = "") -> str:
    """Encrypts text into Polybius coordinate pairs. J is folded into I."""
    letter_to_coords, _ = _build_grid(keyword)

    result = ""
    for char in text.upper():
        if char == "J":
            char = "I"
        if char in letter_to_coords:
            result += letter_to_coords[char]
        else:
            result += char
    return result


def decrypt_polybius(text: str, keyword: str = "") -> str:
    """Decrypts a Polybius digit stream back into letters."""
    _, coords_to_letter = _build_grid(keyword)

    result = ""
    i = 0
    while i < len(text):
        pair = text[i:i + 2]
        if len(pair) == 2 and pair.isdigit() and pair in coords_to_letter:
            result += coords_to_letter[pair]
            i += 2
        else:
            result += text[i]
            i += 1
    return result
