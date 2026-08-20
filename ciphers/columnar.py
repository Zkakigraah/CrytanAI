"""
Columnar Transposition Cipher.

Writes the plaintext into a grid, row by row, under a keyword (the
keyword's length sets the number of columns); reads the ciphertext out
column by column, in the order the keyword's letters would sort
alphabetically. E.g. keyword "ZEBRA" sorts to A-B-E-R-Z, so the column
under the 'A' (position 4) is read first, then the column under 'B'
(position 2), and so on.

This implementation uses irregular columns (the last row is simply left
short) rather than padding with a filler character -- that keeps
encrypt/decrypt an exact inverse of each other with no synthetic
characters added, at the cost of a little extra bookkeeping on decrypt to
know how long each column is.

Every character, including case, spaces, and punctuation, is treated as a
symbol to be reordered -- this cipher doesn't care what alphabet it's
operating on, only where each symbol sits in the grid.
"""


def _column_order(keyword: str) -> list[int]:
    """Returns the original column indices in the order they should be
    read: alphabetically by keyword letter, ties broken by original
    left-to-right position."""
    return sorted(range(len(keyword)), key=lambda i: (keyword[i].upper(), i))


def encrypt_columnar(text: str, keyword: str) -> str:
    """Encrypts text using columnar transposition under the given keyword."""
    if not keyword or not keyword.isalpha():
        raise ValueError("Columnar keyword must be a non-empty alphabetic string.")

    num_cols = len(keyword)
    order = _column_order(keyword)

    columns = [[] for _ in range(num_cols)]
    for i, ch in enumerate(text):
        columns[i % num_cols].append(ch)

    return "".join("".join(columns[col_idx]) for col_idx in order)


def decrypt_columnar(text: str, keyword: str) -> str:
    """Decrypts columnar-transposition ciphertext under the given keyword."""
    if not keyword or not keyword.isalpha():
        raise ValueError("Columnar keyword must be a non-empty alphabetic string.")

    num_cols = len(keyword)
    n = len(text)
    order = _column_order(keyword)

    base_len = n // num_cols
    remainder = n % num_cols
    # The first `remainder` columns (by original left-to-right position)
    # are one character longer, since row-by-row writing fills them
    # first on the final, partial row.
    col_lengths = [base_len + 1 if col < remainder else base_len for col in range(num_cols)]

    columns = [None] * num_cols
    pos = 0
    for col_idx in order:
        length = col_lengths[col_idx]
        columns[col_idx] = text[pos:pos + length]
        pos += length

    result = []
    col_pointers = [0] * num_cols
    for row in range(base_len + 1):
        for col in range(num_cols):
            if row < col_lengths[col]:
                result.append(columns[col][col_pointers[col]])
                col_pointers[col] += 1
    return "".join(result)
