"""Statistical (count-based) n-gram language models.

Self-contained: no imports outside the standard library, so this file can be
dropped into any project layout.

Four smoothing regimes share one counting implementation, so any difference in
results is attributable solely to how UNSEEN events are handled:

    "mle"      maximum likelihood. Unseen n-gram -> probability 0, hence
               infinite perplexity. The failure mode this work exists to study.
    "laplace"  add-k. Every unseen n-gram receives the same mass regardless of
               plausibility, so the model cannot discriminate among them.
    "backoff"  Katz-style backoff with absolute discounting. Consults the
               lower order ONLY when the n-gram was never observed.
    "kn"       Interpolated Kneser-Ney. ALWAYS mixes orders, and the lower
               order is estimated from context diversity rather than raw
               frequency.

Usage
-----
    from ngram import NgramLM

    lm = NgramLM(order=3, vocab_size=len(vocab), bos=BOS_ID, eos=EOS_ID,
                 smoothing="kn").fit(encoded_train)

    lm.token_logprobs(encoded_sentence)   # -> list of natural-log probs
    lm.prob((w1, w2, w3))                 # -> P(w3 | w1 w2)

Token accounting: a sentence ``w1 ... wT`` yields ``T + 1`` predictions
(the tokens plus ``</s>``). Contexts at the start are padded with
``order - 1`` copies of ``<s>``.
"""

from __future__ import annotations

import math
import time
from collections import Counter, defaultdict
from typing import Iterable

__all__ = ["NgramLM", "ngrams", "DEFAULT_LAPLACE_K", "DEFAULT_KN_DISCOUNT"]

DEFAULT_LAPLACE_K = 1.0
DEFAULT_KN_DISCOUNT = 0.75


def ngrams(ids: list[int], order: int, bos: int, eos: int
           ) -> Iterable[tuple[int, ...]]:
    """Yield padded n-grams for one encoded sentence.

    ``[w1, w2]`` with order 3 yields ``(<s>, <s>, w1)``, ``(<s>, w1, w2)``,
    ``(w1, w2, </s>)`` -- three predictions for a two-token sentence.
    """
    padded = [bos] * (order - 1) + list(ids) + [eos]
    for i in range(order - 1, len(padded)):
        yield tuple(padded[i - order + 1:i + 1])


class NgramLM:
    """N-gram language model with selectable smoothing."""

    def __init__(self, order: int, vocab_size: int, bos: int, eos: int,
                 smoothing: str = "mle", k: float = DEFAULT_LAPLACE_K,
                 discount: float = DEFAULT_KN_DISCOUNT):
        if smoothing not in {"mle", "laplace", "backoff", "kn"}:
            raise ValueError(f"unknown smoothing: {smoothing}")
        self.order = order
        self.V = vocab_size
        self.bos = bos
        self.eos = eos
        self.smoothing = smoothing
        self.k = k
        self.d = discount
        self.train_seconds = 0.0

        self.ngram_counts: list[Counter] = [Counter() for _ in range(order + 1)]
        self.context_totals: list[Counter] = [Counter() for _ in range(order + 1)]
        # Kneser-Ney continuation statistics
        self.cont_following: list[dict[tuple, set]] = [defaultdict(set) for _ in range(order + 1)]
        self.cont_preceding: list[dict[tuple, set]] = [defaultdict(set) for _ in range(order + 1)]
        self.cont_total: list[int] = [0] * (order + 1)
        self.unigram_total = 0
        self._alpha_cache: dict[tuple[int, tuple], float] = {}

    @property
    def name(self) -> str:
        label = {"mle": "MLE", "laplace": f"add-{self.k:g}",
                 "backoff": "backoff", "kn": "KN"}[self.smoothing]
        return f"{self.order}-gram {label}"

    # ----------------------------------------------------------------- fit
    def fit(self, encoded: list[list[int]]) -> "NgramLM":
        """Accumulate counts for every order up to ``self.order``."""
        t0 = time.perf_counter()
        for n in range(1, self.order + 1):
            for ids in encoded:
                for g in ngrams(ids, n, self.bos, self.eos):
                    self.ngram_counts[n][g] += 1
                    ctx = g[:-1]
                    self.context_totals[n][ctx] += 1
                    if n > 1:
                        self.cont_following[n][ctx].add(g[-1])
                        self.cont_preceding[n][g[1:]].add(g[0])
            if n > 1:
                self.cont_total[n] = len(self.ngram_counts[n])
        self.unigram_total = sum(self.ngram_counts[1].values())
        self._alpha_cache.clear()
        self.train_seconds = time.perf_counter() - t0
        return self

    # --------------------------------------------------------------- score
    def _p_mle(self, g: tuple[int, ...]) -> float:
        n = len(g)
        denom = self.context_totals[n][g[:-1]]
        return self.ngram_counts[n][g] / denom if denom else 0.0

    def _p_laplace(self, g: tuple[int, ...]) -> float:
        n = len(g)
        num = self.ngram_counts[n][g] + self.k
        denom = self.context_totals[n][g[:-1]] + self.k * self.V
        return num / denom

    # ------------------------------------------------------------- backoff
    def _p_backoff(self, g: tuple[int, ...]) -> float:
        """Katz-style backoff with absolute discounting.

                        max(c(h,w) - d, 0) / c(h)        if c(h,w) > 0
        P_bo(w | h) =
                        alpha(h) * P_bo(w | h')          otherwise

        Unlike interpolated Kneser-Ney this consults the lower order ONLY
        when the higher-order n-gram was never observed.
        """
        n = len(g)
        if n == 1:
            # Add-one unigram base so that every word id retains mass.
            return (self.ngram_counts[1][g] + 1.0) / (self.unigram_total + self.V)

        ctx = g[:-1]
        denom = self.context_totals[n][ctx]
        if denom == 0:
            # Context itself unseen: nothing to discount, defer completely.
            return self._p_backoff(g[1:])

        count = self.ngram_counts[n][g]
        if count > 0:
            return max(count - self.d, 0.0) / denom
        return self._alpha(n, ctx) * self._p_backoff(g[1:])

    def _alpha(self, n: int, ctx: tuple[int, ...]) -> float:
        """Backoff weight: discounted mass / lower-order mass still available.

        beta(h) is the probability mass freed by discounting. It must be
        redistributed over exactly those words NOT seen after h, so it is
        divided by the lower-order probability of that unseen set.
        """
        key = (n, ctx)
        cached = self._alpha_cache.get(key)
        if cached is not None:
            return cached

        denom = self.context_totals[n][ctx]
        seen = self.cont_following[n].get(ctx, ())
        beta = self.d * len(seen) / denom if denom else 1.0

        seen_lower_mass = sum(self._p_backoff(ctx[1:] + (w,)) for w in seen)
        remaining = 1.0 - seen_lower_mass
        alpha = beta / remaining if remaining > 1e-12 else 0.0

        self._alpha_cache[key] = alpha
        return alpha

    # ------------------------------------------------------- Kneser-Ney
    def _p_kn(self, g: tuple[int, ...]) -> float:
        """Interpolated Kneser-Ney, recursive on n-gram order."""
        n = len(g)
        if n == 1:
            # Unigram base: continuation probability -- how many distinct
            # contexts does this word follow? Smoothed uniformly so that
            # unseen word ids still receive mass.
            total = self.cont_total[2]
            if total:
                distinct = len(self.cont_preceding[2].get(g, ()))
                return (distinct + 1) / (total + self.V)
            return 1.0 / self.V

        ctx = g[:-1]
        denom = self.context_totals[n][ctx]
        lower = self._p_kn(g[1:])
        if denom == 0:
            return lower
        count = self.ngram_counts[n][g]
        first = max(count - self.d, 0.0) / denom
        n_following = len(self.cont_following[n].get(ctx, ()))
        lam = (self.d * n_following) / denom
        return first + lam * lower

    # ---------------------------------------------------------- interface
    def prob(self, g: tuple[int, ...]) -> float:
        """P(last token | preceding tokens) under the selected smoothing."""
        if self.smoothing == "mle":
            return self._p_mle(g)
        if self.smoothing == "laplace":
            return self._p_laplace(g)
        if self.smoothing == "backoff":
            return self._p_backoff(g)
        return self._p_kn(g)

    def sentence_logprob(self, ids: list[int]) -> tuple[float, int]:
        """(natural-log probability of the sentence, number of predictions).

        Returns -inf if any n-gram receives zero probability, which is the
        expected behaviour for unsmoothed maximum likelihood.
        """
        total = 0.0
        n_pred = 0
        for g in ngrams(ids, self.order, self.bos, self.eos):
            p = self.prob(g)
            n_pred += 1
            if p <= 0.0:
                return float("-inf"), len(ids) + 1
            total += math.log(p)
        return total, n_pred

    def token_logprobs(self, ids: list[int]) -> list[float]:
        """Log probability of each prediction, aligned with [w1 ... wT, </s>]."""
        out = []
        for g in ngrams(ids, self.order, self.bos, self.eos):
            p = self.prob(g)
            out.append(math.log(p) if p > 0.0 else float("-inf"))
        return out

    # ---------------------------------------------------------------- misc
    def n_parameters(self) -> int:
        """Number of stored n-gram counts -- the model's memory footprint."""
        return sum(len(c) for c in self.ngram_counts[1:])

    def check_normalisation(self, ctx: tuple[int, ...]) -> float:
        """sum_w P(w | ctx) over the whole vocabulary; should be 1.0.

        A smoothing scheme is only valid if it normalises. This is the
        cheapest way to catch mass that is discounted but never redistributed.
        """
        return sum(self.prob(ctx + (w,)) for w in range(self.V))
