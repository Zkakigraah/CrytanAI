"""
Scytale Cipher.

One of the oldest known transposition ciphers (ancient Sparta): a strip
of parchment is wrapped around a rod of a given circumference and the
message is written down its length; unwrapped, the letters appear
scrambled. Mathematically this is equivalent to writing the plaintext
into a grid column by column (`rows` = the rod's circumference in
characters per wrap) and reading it back out row by row.

Scytale is really a columnar transposition with a trivial, fixed
(left-to-right, unkeyed) column order -- but it's implemented completely
independently here, both for historical clarity (it predates
keyword-based columnar transposition by centuries) and to keep every
cipher module self-contained.

Every character, including case, spaces, and punctuation, is treated as a
symbol to be reordered.
"""
import math


def encrypt_scytale(text: str, rows: int) -> str:
    """Encrypts text by writing it down columns of height `rows` and
    reading the result back out across rows."""
    if rows < 1:
        raise ValueError("Scytale rows must be a positive integer.")
    if rows >= len(text):
        return text

    num_cols = math.ceil(len(text) / rows)
    grid = [[None] * num_cols for _ in range(rows)]

    for i, ch in enumerate(text):
        r, c = i % rows, i // rows
        grid[r][c] = ch

    result = []
    for r in range(rows):
        for c in range(num_cols):
            if grid[r][c] is not None:
                result.append(grid[r][c])
    return "".join(result)


def decrypt_scytale(text: str, rows: int) -> str:
    """Decrypts scytale ciphertext produced with the given number of rows."""
    if rows < 1:
        raise ValueError("Scytale rows must be a positive integer.")
    if rows >= len(text):
        return text

    n = len(text)
    num_full_cols = n // rows
    remainder = n % rows
    row_lengths = [num_full_cols + 1 if r < remainder else num_full_cols for r in range(rows)]

    rows_of_chars = []
    pos = 0
    for r in range(rows):
        length = row_lengths[r]
        rows_of_chars.append(text[pos:pos + length])
        pos += length

    num_cols = num_full_cols + (1 if remainder else 0)
    grid = [[None] * num_cols for _ in range(rows)]
    for r in range(rows):
        for c, ch in enumerate(rows_of_chars[r]):
            grid[r][c] = ch

    result = []
    for c in range(num_cols):
        for r in range(rows):
            if grid[r][c] is not None:
                result.append(grid[r][c])
    return "".join(result)
