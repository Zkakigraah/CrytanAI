"""
Playfair Cipher Solver.

Best-effort grid hill-climbing: the search space is arrangements of 25
letters in a 5x5 grid (25! possible grids -- similar magnitude to
substitution's 26!), so this uses the same coordinate-ascent idea as
solvers/substitution_solver.py: swap two letters' grid positions, keep
the swap if the resulting decryption scores higher, run many random
restarts -- just applied to grid position instead of a flat cipher
alphabet. A candidate grid, passed as if it were a 25-letter "keyword",
reconstructs that exact grid via utils.text_utils.build_keyed_alphabet
(a full-length permutation has no remaining letters left to append), so
no changes to ciphers/playfair.py are needed to drive this search.

Once a candidate grid decrypts the ciphertext back to a plain letter
stream, that stream is scored with the exact same quadgram model used
everywhere else in this project -- Playfair's digraph structure only
matters for the encrypt/decrypt transform itself, not for scoring its
output.

This is flagged best_effort, not high confidence, and that's a real,
tested finding, not a generic caveat: on a ~280-character ciphertext, the
true grid scores clearly higher than anything this search finds (-4.7 vs.
-6.8, even after 5x the default compute budget), and switching from
greedy hill-climbing to simulated annealing didn't close that gap either.
A single grid-position swap changes several digraph mappings at once (a
rectangle-rule swap touches every pair involving either letter), which
makes the fitness landscape rough enough that simple local search
struggles to find the true grid -- consistent with Playfair's actual
history: it resisted casual cryptanalysis for decades longer than simple
substitution precisely because digraph substitution obscures single-
letter structure. A reliable ciphertext-only break needs a smarter
technique (digraph-frequency-specific methods, e.g. bigram-difference
tables) -- noted as a real extension opportunity, not implemented here.
"""
import random

from ciphers.playfair import decrypt_playfair
from ml.nlp_scorer import default_scorer
from solvers.base import SolveResult
from utils.text_utils import build_keyed_alphabet

RESTARTS = 20
ITERATIONS_PER_RESTART = 800


def _hill_climb(ciphertext: str, iterations: int) -> str:
    grid = list(build_keyed_alphabet("", merge_j_into_i=True))  # the 25-letter base alphabet
    random.shuffle(grid)

    current_score = default_scorer.quick_score(decrypt_playfair(ciphertext, "".join(grid)))
    best_grid, best_score = "".join(grid), current_score

    for _ in range(iterations):
        i, j = random.sample(range(25), 2)
        grid[i], grid[j] = grid[j], grid[i]

        candidate = "".join(grid)
        score = default_scorer.quick_score(decrypt_playfair(ciphertext, candidate))

        if score >= current_score:
            current_score = score
            if score > best_score:
                best_grid, best_score = candidate, score
        else:
            grid[i], grid[j] = grid[j], grid[i]  # undo

    return best_grid


def solve_playfair(ciphertext: str, top_k: int = 3) -> list[SolveResult]:
    """Hill-climbs the 5x5 grid over many random restarts and returns
    the top_k distinct-plaintext results, always at best_effort
    confidence (see module docstring)."""
    found: dict[str, str] = {}
    for _ in range(RESTARTS):
        grid = _hill_climb(ciphertext, ITERATIONS_PER_RESTART)
        plaintext = decrypt_playfair(ciphertext, grid)
        found[plaintext] = grid

    scored = [(pt, grid, default_scorer.score(pt)) for pt, grid in found.items()]
    scored.sort(key=lambda t: -t[2])

    return [
        SolveResult(key=grid, plaintext=pt, score=score, method="hill_climbing", confidence="best_effort")
        for pt, grid, score in scored[:top_k]
    ]
