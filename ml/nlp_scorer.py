"""
The "English Judge": combines several independent statistical signals
about a piece of text into one composite score of "how English does this
look?". This is the single most-used piece of infrastructure in the
project -- every solver's hill-climbing/simulated-annealing search is
guided by this score, and it also feeds the cipher detector's feature
vector.

Structurally, this follows a simple, standard shape: a handful of
sub-scores (letter frequency, bigrams, quadgrams, word-dictionary
matches) combined into one weighted total. What matters is what's behind
those sub-scores: every frequency table here is measured from ~2.6M words
of real public-domain English text (see utils/language_stats.py and
scripts/build_language_model.py), not a short hand-typed list -- and
there's a quadgram layer, which is what actually makes hill-climbing
substitution solvers converge reliably. A scorer built only from single
letters and a couple dozen common bigrams gives a flat, noisy gradient
that local search can easily get stuck on; quadgram statistics are the
standard fix, used by essentially every serious substitution-cipher
solver.
"""
from dataclasses import dataclass

from utils.language_stats import (
    chi_squared_statistic,
    bigram_log_likelihood,
    quadgram_log_likelihood,
    dictionary_word_ratio,
)


@dataclass
class ScoreBreakdown:
    """Every sub-signal that went into a total score, kept around so the
    UI can show *why* a candidate was judged more or less English-like,
    rather than just a single opaque number."""
    chi_squared: float   # distance from expected English letter freq; lower = more English-like
    bigram_ll: float     # avg log10-prob of letter bigrams; closer to 0 = more English-like
    quadgram_ll: float   # avg log10-prob of letter quadgrams; closer to 0 = more English-like
    word_ratio: float    # fraction of tokens that are real English words, 0-1
    total: float          # combined score; higher = more English-like


class EnglishScorer:
    """
    Scores how English-like a piece of text is.

    Two entry points, for two different jobs:

    - `quick_score`: a single fast signal (quadgram log-likelihood) for
      hot loops -- hill-climbing and simulated-annealing solvers call
      this thousands of times per run, so it skips computing the other
      three signals.
    - `full_score` / `score`: every signal combined, for anything that
      isn't a search hot loop -- ranking finished candidates, the cipher
      detector's features, the UI's confidence display.
    """

    # Weights for combining sub-scores in `full_score`. Quadgrams get the
    # largest share because they're the most discriminative single
    # signal (see module docstring); chi-squared the smallest, since it's
    # the coarsest (it only sees the letter distribution, not order).
    CHI2_WEIGHT = 0.15
    BIGRAM_WEIGHT = 0.20
    QUADGRAM_WEIGHT = 0.40
    WORD_WEIGHT = 0.25

    def quick_score(self, text: str) -> float:
        """Fast, single-signal score for search hot loops. Higher is
        more English-like."""
        return quadgram_log_likelihood(text)

    def full_score(self, text: str) -> ScoreBreakdown:
        """Full multi-signal score. Higher `total` is more English-like."""
        chi2 = chi_squared_statistic(text)
        bigram_ll = bigram_log_likelihood(text)
        quad_ll = quadgram_log_likelihood(text)
        word_ratio = dictionary_word_ratio(text)

        # chi-squared is unbounded and "lower is better" -- fold it into
        # a bounded (0, 1], "higher is better" term before weighting.
        chi2_term = 1.0 / (1.0 + chi2 / 50.0)
        # Log-likelihoods are negative, roughly in [-9, 0] in practice
        # (see utils/language_stats.py's floor values) -- rescale to
        # roughly [0, 1] so no single term dominates by magnitude alone.
        bigram_term = max(0.0, (bigram_ll + 9) / 9)
        quadgram_term = max(0.0, (quad_ll + 9) / 9)

        total = (
            self.CHI2_WEIGHT * chi2_term
            + self.BIGRAM_WEIGHT * bigram_term
            + self.QUADGRAM_WEIGHT * quadgram_term
            + self.WORD_WEIGHT * word_ratio
        )

        return ScoreBreakdown(
            chi_squared=chi2, bigram_ll=bigram_ll, quadgram_ll=quad_ll,
            word_ratio=word_ratio, total=total,
        )

    def score(self, text: str) -> float:
        """Convenience: just the combined total from full_score(). This
        is the method most callers outside a search hot loop want."""
        return self.full_score(text).total


# Module-level singleton -- every solver shares one instance rather than
# constructing its own (the class holds no per-call state, so this is
# just avoiding pointless repeated construction).
default_scorer = EnglishScorer()
