"""DiffusionGemma sampling: Algorithm 1 of the technical report (arXiv 2608.00146, p. 10), and a plain
one-token sampler.  Only numpy; no model.  The model is passed in as two callables, so the tests can use fixed logits.

    decode(canvas_ids [C], z) -> logits [C, V]      z is None on the first step (the report's z1 = 0)
    self_cond(p [C, V]) -> z                        what the next decode call takes as z

`temperature` scales the schedule: tau runs from TAU_MAX * s to TAU_MIN * s with s = temperature / 0.8, so the
default temperature 0.8 is the report's 0.8 -> 0.4.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

CANVAS = 256
MAX_STEPS = 48
ENTROPY_BOUND = 0.1
STOP_ENTROPY = 0.005
TAU_MAX = 0.8
TAU_MIN = 0.4
DEFAULT_TEMPERATURE = 0.8


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def entropy(p: np.ndarray) -> np.ndarray:
    """Per position entropy of p [C, V], in nats."""
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(p > 0, p * np.log(p), 0.0)
    return -t.sum(axis=-1)


def tau_at(n: int, steps: int, temperature: float = DEFAULT_TEMPERATURE,
           tau_max: float = TAU_MAX, tau_min: float = TAU_MIN) -> float:
    """Temperature of step n (1-based): t = 1 - (n-1)/N, tau = (tau_max - tau_min) * t + tau_min, both scaled."""
    t = 1.0 - (n - 1) / steps
    s = max(float(temperature), 1e-3) / DEFAULT_TEMPERATURE
    return ((tau_max - tau_min) * t + tau_min) * s


def accept_count(ent_sorted: np.ndarray, bound: float) -> int:
    """k = the largest m such that the entropies of the first m-1 positions add up to at most `bound`
    (at least 1, at most C).  `ent_sorted` is ascending."""
    c = np.cumsum(ent_sorted)
    return int(min(len(ent_sorted), 1 + int((c <= bound).sum())))


def sample_rows(p: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One draw per row of p [C, V]."""
    cdf = np.cumsum(p, axis=-1)
    u = rng.random((p.shape[0], 1)) * cdf[:, -1:]
    return np.minimum((cdf < u).sum(axis=-1), p.shape[-1] - 1).astype(np.int64)


@dataclass
class CanvasResult:
    tokens: np.ndarray        # the argmax canvas [C]
    steps: int                # denoising steps run (decode calls)
    stopped_early: bool       # the adaptive stop fired


def denoise_canvas(decode: Callable, self_cond: Callable, rng: np.random.Generator, vocab: int,
                   canvas: int = CANVAS, max_steps: int = MAX_STEPS, entropy_bound: float = ENTROPY_BOUND,
                   stop_entropy: float = STOP_ENTROPY, temperature: float = DEFAULT_TEMPERATURE,
                   on_step: Optional[Callable[[int, np.ndarray], None]] = None) -> CanvasResult:
    """Algorithm 1.  `on_step(n, argmax_canvas)` is called after every step (also the last)."""
    x = rng.integers(0, vocab, size=canvas, dtype=np.int64)       # x1 ~ Uniform(V)
    z = None
    prev = None
    idx = np.arange(canvas)
    x0 = x
    for n in range(1, max_steps + 1):
        logits = np.asarray(decode(x, z), dtype=np.float32)
        x0 = logits.argmax(axis=-1).astype(np.int64)
        tau = tau_at(n, max_steps, temperature)
        p = softmax(logits / tau)
        e = entropy(p)
        if on_step is not None:
            on_step(n, x0)
        if n > 1 and float(e.mean()) <= stop_entropy and prev is not None and np.array_equal(x0, prev):
            return CanvasResult(x0, n, True)
        prev = x0
        if n < max_steps:
            z = self_cond(p)
            xd = sample_rows(p, rng)
            order = np.lexsort((idx, e))                          # entropy ascending, ties by index
            k = accept_count(e[order], entropy_bound)
            nxt = rng.integers(0, vocab, size=canvas, dtype=np.int64)   # renoise everything else
            keep = order[:k]
            nxt[keep] = xd[keep]
            x = nxt
    return CanvasResult(x0, max_steps, False)


def sample_token(logits: np.ndarray, rng: np.random.Generator, temperature: float = 0.0, top_k: int = 0,
                 top_p: float = 1.0, min_p: float = 0.0, history: Optional[list] = None,
                 penalty_repeat: float = 1.0, penalty_freq: float = 0.0, penalty_present: float = 0.0) -> int:
    """One-token mode: normal temperature / top_k / top_p / min_p sampling of logits [V].  Penalties count over
    `history` (the caller passes the window)."""
    l = np.asarray(logits, dtype=np.float64).copy()
    if history and (penalty_repeat != 1.0 or penalty_freq or penalty_present):
        ids, counts = np.unique(np.asarray(history, dtype=np.int64), return_counts=True)
        ok = (ids >= 0) & (ids < len(l))
        ids, counts = ids[ok], counts[ok]
        v = l[ids]
        v = np.where(v > 0, v / penalty_repeat, v * penalty_repeat)
        l[ids] = v - penalty_freq * counts - penalty_present
    if temperature <= 0.0:
        return int(l.argmax())
    l = l / temperature
    if top_k and 0 < top_k < len(l):
        cut = np.partition(l, -top_k)[-top_k]
        l[l < cut] = -np.inf
    p = softmax(l)
    if min_p > 0.0:
        p = np.where(p >= min_p * p.max(), p, 0.0)
        p /= p.sum()
    if 0.0 < top_p < 1.0:
        o = np.argsort(-p, kind="stable")
        c = np.cumsum(p[o])
        keep = o[: int((c < top_p).sum()) + 1]
        q = np.zeros_like(p)
        q[keep] = p[keep]
        p = q / q.sum()
    return int(sample_rows(p[None, :], rng)[0])
