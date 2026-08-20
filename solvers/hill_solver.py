"""
Hill Cipher Solver.

Deliberately NOT a ciphertext-only cracker -- see ciphers/hill.py and this
project's PROJECT_QA.md for the full reasoning. In short: Hill's
block/matrix structure doesn't leave the kind of single-symbol
statistical residue that frequency analysis (the technique behind every
other solver in this project) needs. A block of n letters maps to another
block of n letters via matrix multiplication mod 26, mixing every letter
in the block together -- there's no "most common symbol" to anchor a
guess the way there is for single-letter or even digraph substitution,
and the alphabet size for a block (26^n possible blocks) grows too fast
for frequency statistics to stay reliable past n=1.

Real-world Hill cryptanalysis is instead a KNOWN-PLAINTEXT attack: if you
have (or can plausibly guess) n blocks of n known plaintext letters
aligned to a known position in the ciphertext -- a "crib", classically
something like a predictable greeting or letterhead -- recovering the key
is a clean, exact piece of linear algebra:

    C = K * P  (mod 26)   =>   K = C * P^-1  (mod 26)

where C and P are the n x n matrices built from the aligned
ciphertext/plaintext blocks (P must be invertible mod 26 -- not every
crib works, which mirrors a real constraint of the attack, not an
implementation gap). This module provides that crib attack, plus a thin
pass-through for decrypting with an already-known key -- the other
realistic way this cipher is actually used.
"""
from sympy import Matrix

from ciphers.hill import decrypt_hill
from ml.nlp_scorer import default_scorer
from solvers.base import SolveResult
from utils.text_utils import ALPHABET, clean_letters_only


def decrypt_known_key(ciphertext: str, key: str) -> SolveResult:
    """Decrypts with an already-known key. Not a search -- just wraps
    the result in the same SolveResult shape every other solver uses, so
    the UI can treat all ciphers uniformly."""
    plaintext = decrypt_hill(ciphertext, key)
    return SolveResult(key=key, plaintext=plaintext, score=default_scorer.score(plaintext),
                        method="known_key", confidence="high")


def crack_known_plaintext(
    ciphertext: str, known_plaintext: str, block_size: int, ciphertext_offset: int = 0,
) -> SolveResult:
    """
    Recovers the Hill key from a known-plaintext crib aligned at
    `ciphertext_offset` letters into the ciphertext (default: the very
    start -- the classic crib-attack setup, e.g. a known greeting or
    letterhead; pass a different offset if the known fragment sits
    elsewhere, e.g. a predictable closing line). Needs at least
    block_size^2 letters of crib.

    Raises ValueError if the crib is too short, if the offset isn't
    block-aligned, or if the crib's plaintext matrix isn't invertible mod
    26 (not every crib works for a given block size -- try a different
    crib, position, or length).
    """
    n = block_size
    if ciphertext_offset % n != 0:
        raise ValueError(
            f"ciphertext_offset ({ciphertext_offset}) must be a multiple of block_size "
            f"({n}): Hill encrypts in fixed-size blocks starting from position 0, so a "
            f"crib that doesn't start on a block boundary doesn't correspond to whole "
            f"blocks and can't be used this way -- the math would silently give a "
            f"meaningless result rather than fail loudly, which is worse, so this checks "
            f"for it explicitly."
        )
    crib_letters = clean_letters_only(known_plaintext)
    cipher_letters = clean_letters_only(ciphertext)

    if len(crib_letters) < n * n:
        raise ValueError(
            f"Need at least {n * n} letters of known plaintext for a {n}x{n} key "
            f"(got {len(crib_letters)})."
        )
    if len(cipher_letters) < ciphertext_offset + n * n:
        raise ValueError("Ciphertext is shorter than the crib's offset plus required length.")

    p_nums = [ALPHABET.index(c) for c in crib_letters[:n * n]]
    c_nums = [ALPHABET.index(c) for c in cipher_letters[ciphertext_offset:ciphertext_offset + n * n]]

    # Column k of each matrix is block k (n consecutive letters, in
    # original order) -- matching exactly how ciphers/hill.py turns a
    # run of n letters into a column vector for one block.
    P = Matrix(n, n, lambda r, col: p_nums[col * n + r])
    C = Matrix(n, n, lambda r, col: c_nums[col * n + r])

    try:
        P_inv = P.inv_mod(26)
    except Exception as e:
        raise ValueError(
            "This crib's plaintext matrix isn't invertible mod 26, so it can't "
            "be used to solve for the key -- try a different or longer crib."
        ) from e

    K = (C * P_inv) % 26

    # ciphers/hill.py's _get_key_matrix builds the matrix ROW-MAJOR from
    # the flat key string (row i = key[i*n:(i+1)*n]), so reconstructing
    # the string has to iterate row-outer, column-inner to match -- the
    # reverse order would silently produce a transposed, wrong key.
    key = "".join(ALPHABET[int(K[r, col]) % 26] for r in range(n) for col in range(n))

    plaintext = decrypt_hill(ciphertext, key)
    return SolveResult(key=key, plaintext=plaintext, score=default_scorer.score(plaintext),
                        method="known_plaintext_crib", confidence="high")
