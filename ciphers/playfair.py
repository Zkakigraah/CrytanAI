"""
Playfair Cipher.

A digraph substitution cipher (Charles Wheatstone, 1854) that encrypts
pairs of letters at a time using a 5x5 keyed grid (I/J merged, same grid
construction as polybius.py -- both build theirs via
utils.text_utils.build_keyed_alphabet, so this file and polybius.py never
import each other, only the shared utility).

Because it operates on letter PAIRS with grid-geometry rules, Playfair
can't preserve inline punctuation/spacing the way the monoalphabetic and
transposition ciphers in this project do -- the classical convention
(and the one used here) is to strip to letters only, split into digraphs,
and separate any doubled letters within a pair with a filler 'X'. This is
authentic Playfair behavior, not a shortcut: real Playfair ciphertext was
always transmitted as uniform letter groups regardless of the original
message's spacing.
"""
from utils.text_utils import build_keyed_alphabet

GRID_SIZE = 5
FILLER = "X"


def _build_grid(keyword: str):
    alphabet_25 = build_keyed_alphabet(keyword, merge_j_into_i=True)
    pos = {ch: divmod(i, GRID_SIZE) for i, ch in enumerate(alphabet_25)}
    grid = [[alphabet_25[r * GRID_SIZE + c] for c in range(GRID_SIZE)] for r in range(GRID_SIZE)]
    return grid, pos


def _prepare_digraphs(text: str) -> list[str]:
    """Cleans text to letters only (J->I), then splits into digraphs,
    inserting a filler between repeated letters in a pair and padding a
    final leftover letter."""
    letters = [("I" if ch == "J" else ch) for ch in text.upper() if ch.isalpha()]

    digraphs = []
    i = 0
    while i < len(letters):
        a = letters[i]
        if i + 1 < len(letters):
            b = letters[i + 1]
            if a == b:
                digraphs.append(a + FILLER)
                i += 1
            else:
                digraphs.append(a + b)
                i += 2
        else:
            digraphs.append(a + FILLER)
            i += 1
    return digraphs


def encrypt_playfair(text: str, keyword: str = "") -> str:
    """Encrypts text using the Playfair cipher. Output is letters-only,
    grouped into space-separated digraphs."""
    grid, pos = _build_grid(keyword)
    digraphs = _prepare_digraphs(text)

    out_pairs = []
    for pair in digraphs:
        a, b = pair[0], pair[1]
        ra, ca = pos[a]
        rb, cb = pos[b]

        if ra == rb:
            out_pairs.append(grid[ra][(ca + 1) % GRID_SIZE] + grid[rb][(cb + 1) % GRID_SIZE])
        elif ca == cb:
            out_pairs.append(grid[(ra + 1) % GRID_SIZE][ca] + grid[(rb + 1) % GRID_SIZE][cb])
        else:
            out_pairs.append(grid[ra][cb] + grid[rb][ca])

    return " ".join(out_pairs)


def decrypt_playfair(text: str, keyword: str = "") -> str:
    """Decrypts Playfair ciphertext (space-separated digraphs or not --
    both are accepted, since decryption only reads the letters)."""
    grid, pos = _build_grid(keyword)
    letters = [ch for ch in text.upper() if ch.isalpha()]

    out = []
    for i in range(0, len(letters) - 1, 2):
        a, b = letters[i], letters[i + 1]
        ra, ca = pos[a]
        rb, cb = pos[b]

        if ra == rb:
            out.append(grid[ra][(ca - 1) % GRID_SIZE] + grid[rb][(cb - 1) % GRID_SIZE])
        elif ca == cb:
            out.append(grid[(ra - 1) % GRID_SIZE][ca] + grid[(rb - 1) % GRID_SIZE][cb])
        else:
            out.append(grid[ra][cb] + grid[rb][ca])

    return "".join(out)
