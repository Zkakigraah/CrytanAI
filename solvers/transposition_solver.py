"""
Transposition Solver.

Covers Rail Fence, Scytale, and Columnar Transposition. Transposition
ciphers only reorder letters -- they never relabel them -- so unigram
letter frequency (the tool behind every monoalphabetic solver in this
project) is completely useless here: a transposed ciphertext has EXACTLY
the same letter frequency distribution as the plaintext, by construction.
What breaks under a wrong reordering is bigram/quadgram structure and
word formation, both of which a correct unscrambling restores and an
incorrect one doesn't -- so everything here leans on the English scorer's
n-gram signals, not chi-squared.

Rail Fence and Scytale each take a single small integer key (rail count /
row count), so both are small, exact brute-force searches -- structurally
identical to Caesar's, just over a different small key space. Columnar
Transposition's key is effectively a column ORDER (a permutation), which
grows factorially with the number of columns: exhaustively tried for
small keywords (exact), and hill-climbed for larger ones (the same
coordinate-ascent idea used by every other search-based solver in this
project, just applied to permutation space instead of substitution-key or
single-symbol space).
"""
import random
from itertools import permutations

from ciphers.railfence import decrypt_railfence
from ciphers.scytale import decrypt_scytale
from ciphers.columnar import decrypt_columnar
from ml.nlp_scorer import default_scorer
from solvers.base import SolveResult, brute_force_rank

MAX_RAILS = 15
MAX_SCYTALE_ROWS = 15

EXHAUSTIVE_PERMUTATION_LIMIT = 7   # 7! = 5040 -- still fast to try every order, and
                                    # verified reliable (see tests/test_solvers.py).
MAX_COLUMNAR_COLUMNS = 12          # beyond EXHAUSTIVE_PERMUTATION_LIMIT, falls back to
                                    # hill-climbing, which is best-effort, not reliable --
                                    # see solve_columnar's docstring.
HILL_CLIMB_RESTARTS = 15
HILL_CLIMB_ITERATIONS = 400


def solve_railfence(ciphertext: str, top_k: int = 3) -> list[SolveResult]:
    """Brute forces every plausible rail count."""
    max_rails = min(MAX_RAILS, max(2, len(ciphertext) - 1))
    return brute_force_rank(
        ciphertext, candidate_keys=range(2, max_rails + 1),
        decrypt_fn=decrypt_railfence, method="brute_force", confidence="high", top_k=top_k,
    )


def solve_scytale(ciphertext: str, top_k: int = 3) -> list[SolveResult]:
    """Brute forces every plausible row count."""
    max_rows = min(MAX_SCYTALE_ROWS, max(2, len(ciphertext) - 1))
    return brute_force_rank(
        ciphertext, candidate_keys=range(2, max_rows + 1),
        decrypt_fn=decrypt_scytale, method="brute_force", confidence="high", top_k=top_k,
    )


def _order_to_keyword(order: tuple) -> str:
    """ciphers/columnar.py's decrypt takes a keyword, not a raw column
    order -- this builds the shortest synthetic keyword (using early
    alphabet letters, each exactly once) whose alphabetical sort
    reproduces `order` exactly. Any keyword with that same relative
    letter order decrypts identically; which letters we pick is
    arbitrary, only their relative order matters."""
    keyword_chars = [""] * len(order)
    for rank, col_idx in enumerate(order):
        keyword_chars[col_idx] = chr(ord("A") + rank)
    return "".join(keyword_chars)


def _score_order(ciphertext: str, order: tuple) -> float:
    return default_scorer.quick_score(decrypt_columnar(ciphertext, _order_to_keyword(order)))


def _exhaustive_columnar(ciphertext: str, num_cols: int) -> str:
    best_order, best_score = None, float("-inf")
    for order in permutations(range(num_cols)):
        score = _score_order(ciphertext, order)
        if score > best_score:
            best_score, best_order = score, order
    return _order_to_keyword(best_order)


def _hill_climb_columnar(ciphertext: str, num_cols: int) -> str:
    best_overall_order, best_overall_score = None, float("-inf")
    for _ in range(HILL_CLIMB_RESTARTS):
        order = list(range(num_cols))
        random.shuffle(order)
        score = _score_order(ciphertext, tuple(order))

        for _ in range(HILL_CLIMB_ITERATIONS):
            i, j = random.sample(range(num_cols), 2)
            order[i], order[j] = order[j], order[i]
            new_score = _score_order(ciphertext, tuple(order))
            if new_score >= score:
                score = new_score
            else:
                order[i], order[j] = order[j], order[i]  # undo

        if score > best_overall_score:
            best_overall_score, best_overall_order = score, tuple(order)

    return _order_to_keyword(best_overall_order)


def solve_columnar(ciphertext: str, top_k: int = 3) -> list[SolveResult]:
    """
    Tries plausible column counts and, for each, finds the best column
    order.

    Up to EXHAUSTIVE_PERMUTATION_LIMIT columns, every possible order is
    tried -- exact, verified reliable. Beyond that, column-order search
    falls back to hill-climbing, which is best-effort: verified directly
    (see this project's dev notes / tests/test_solvers.py) to sometimes
    fail to converge even with a 10x larger search budget than the
    default here, on a 12-column keyword with a few hundred characters of
    ciphertext. That isn't a bug to fix with more compute -- it's a real
    property of searching a permutation space this large (12! is roughly
    479 million) with only a coordinate-ascent swap search and a modest
    per-column sample size. Longer-keyword columnar results are reported
    honestly as "best_effort" confidence rather than papered over.
    """
    max_cols = min(MAX_COLUMNAR_COLUMNS, max(2, len(ciphertext) // 2))

    found: dict[str, tuple[str, float, str]] = {}
    for num_cols in range(2, max_cols + 1):
        if num_cols <= EXHAUSTIVE_PERMUTATION_LIMIT:
            keyword, confidence = _exhaustive_columnar(ciphertext, num_cols), "high"
        else:
            keyword, confidence = _hill_climb_columnar(ciphertext, num_cols), "best_effort"

        plaintext = decrypt_columnar(ciphertext, keyword)
        score = default_scorer.score(plaintext)
        if plaintext not in found or score > found[plaintext][1]:
            found[plaintext] = (keyword, score, confidence)

    ranked = sorted(found.items(), key=lambda kv: -kv[1][1])
    return [
        SolveResult(key=keyword, plaintext=pt, score=score,
                    method="exhaustive" if conf == "high" else "hill_climbing", confidence=conf)
        for pt, (keyword, score, conf) in ranked[:top_k]
    ]
