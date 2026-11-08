"""Models for the DiffusionGemma sidecar engine.

The engine needs this much from a model (class `Model` below):

    vocab_size                    int
    length                        tokens in the cache
    encode(ids, kind)             append ids to the cache; kind is "prompt", "canvas" or "token"
    last_logits() -> [V]          logits of the next token after the cache (one-token mode)
    decode(canvas, z) -> [C, V]   one denoising step over a canvas of ids, with cross-attention to the cache
    self_cond(p) -> z             the self-conditioning input for the next step, from probabilities p [C, V]
    truncate(n)                   keep the first n cache tokens (prefix reuse)
    reset()                       empty the cache

FakeModel is a small numpy model for tests.  MlxModel wraps mlx-vlm.
"""
from __future__ import annotations

import os
import time

import numpy as np


class Model:
    vocab_size: int
    length: int

    def encode(self, ids, kind: str = "prompt") -> None: ...
    def last_logits(self) -> np.ndarray: ...
    def decode(self, canvas, z): ...
    def self_cond(self, p): ...
    def truncate(self, n: int) -> None: ...
    def reset(self) -> None: ...


class FakeModel(Model):
    """A deterministic model that converges.  The reply to a prompt is a fixed token sequence: the prompt's first
    three ids pick it.  Token at reply position i is 4 + (h * 31 + i * 7) % (V - 4); the reply is `reply_len` tokens
    long and then comes the end token (id 1), then filler.  A denoising step puts a peak on the target token of every
    position; the peak grows with the self-conditioning input, so canvases settle in a few steps.
    """

    def __init__(self, vocab: int = 64, reply_len=None, eos: int = 1):
        self.vocab_size = vocab
        self.eos = eos
        self.reply_len = reply_len        # None: 8 + h % 40
        self.cache: list = []
        self.prompt_len = 0
        self.delay = float(os.environ.get("STRATA_FAKE_DELAY_MS", "0")) / 1000.0   # per step, for STOP tests

    @property
    def length(self) -> int:
        return len(self.cache)

    def _h(self) -> int:
        return sum((i + 1) * int(t) for i, t in enumerate(self.cache[:3])) % 9973

    def _rlen(self) -> int:
        return int(self.reply_len) if self.reply_len is not None else 8 + self._h() % 40

    def target(self, pos: int) -> int:
        """The token the reply has at absolute position `pos`."""
        i = pos - self.prompt_len
        if i == self._rlen():
            return self.eos
        return 4 + (self._h() * 31 + i * 7) % (self.vocab_size - 4)

    def encode(self, ids, kind: str = "prompt") -> None:
        self.cache.extend(int(t) for t in ids)
        if kind == "prompt":
            self.prompt_len = len(self.cache)

    def truncate(self, n: int) -> None:
        del self.cache[n:]
        self.prompt_len = min(self.prompt_len, len(self.cache))

    def reset(self) -> None:
        self.cache = []
        self.prompt_len = 0

    def last_logits(self) -> np.ndarray:
        l = np.zeros(self.vocab_size, dtype=np.float32)
        l[self.target(len(self.cache))] = 20.0
        return l

    def decode(self, canvas, z):
        if self.delay:
            time.sleep(self.delay)
        c = len(canvas)
        l = np.zeros((c, self.vocab_size), dtype=np.float32)
        base = len(self.cache)
        for j in range(c):
            t = self.target(base + j)
            conf = 0.0 if z is None else float(z[j, t])
            l[j, t] = 3.0 + 14.0 * conf
        return l

    def self_cond(self, p):
        return p


class MlxModel(Model):
    """DiffusionGemma on MLX through mlx-vlm's `diffusion_gemma` module (mlx-vlm 0.7.x).

    UNTESTED HERE.  Written on Linux against the source of mlx_vlm/models/diffusion_gemma/{diffusion_gemma,language}.py
    and mlx_vlm/generate/diffusion.py (0.7.0).  It has not been run, on a Mac or anywhere else.  Check on a Mac:
      - the calls below exist in the pinned mlx-vlm version (diffusion_prefill_cache, diffusion_update_cache,
        diffusion_decoder_masks, diffusion_decoder_logits, diffusion_self_conditioning, make_cache)
      - the cache objects can be trimmed (`trim`); if not, truncate() falls back to a full re-read
      - one-token mode (last_logits / encode kind "token") uses the encoder's causal path through
        `model(input_ids, cache=...)`; check that this is the path the report's one-token mode uses
    Imports of mlx and mlx_vlm happen here, not at module import, so the rest of the engine runs without them.
    """

    def __init__(self, path: str):
        import mlx.core as mx
        from mlx_vlm import load
        self.mx = mx
        self.model, self.processor = load(path)
        cfg = getattr(self.model.config, "text_config", self.model.config)
        self.vocab_size = int(getattr(cfg, "vocab_size"))
        self.cache = None
        self.ids: list = []
        self._masks = None
        self._masks_len = -1
        self.reset()

    @property
    def length(self) -> int:
        return len(self.ids)

    def reset(self) -> None:
        self.cache = self.model.make_cache()
        self.ids = []
        self._masks = None
        self._masks_len = -1

    def encode(self, ids, kind: str = "prompt") -> None:
        mx = self.mx
        arr = mx.array([[int(t) for t in ids]])
        if kind == "canvas":
            self.cache = self.model.diffusion_update_cache(arr, cache=self.cache)
        elif kind == "prompt":
            self.cache = self.model.diffusion_prefill_cache(
                arr, attention_mask=None, cache=self.cache, pixel_values=None, mm_token_type_ids=None,
                prefill_step_size=None, chunk_prefill=False)
        else:   # one token at a time: the causal path
            self._last = self.model.language_model(arr, cache=self.cache) if hasattr(self.model, "language_model") \
                else self.model(arr, cache=self.cache)
        self.ids.extend(int(t) for t in ids)
        self._masks_len = -1
        mx.eval([c.state for c in self.cache])

    def last_logits(self) -> np.ndarray:
        out = self._last
        logits = getattr(out, "logits", out)
        return np.array(logits[0, -1].astype(self.mx.float32))

    def truncate(self, n: int) -> None:
        drop = len(self.ids) - n
        if drop <= 0:
            return
        if all(hasattr(c, "is_trimmable") and c.is_trimmable() for c in self.cache):
            for c in self.cache:
                c.trim(drop)
            del self.ids[n:]
            self._masks_len = -1
        else:
            raise RuntimeError("the cache cannot be trimmed")

    def decode(self, canvas, z):
        mx = self.mx
        arr = mx.array(np.asarray(canvas, dtype=np.int64)[None, :])
        if self._masks is None or self._masks_len != len(self.ids) or self._masks_canvas != arr.shape[1]:
            self._masks = self.model.diffusion_decoder_masks(arr, self.cache, None)
            self._masks_len, self._masks_canvas = len(self.ids), arr.shape[1]
        logits = self.model.diffusion_decoder_logits(arr, cache=self.cache, self_conditioning=z,
                                                     decoder_attention_mask=self._masks)
        mx.eval(logits)
        return np.array(logits[0].astype(mx.float32))

    def self_cond(self, p):
        mx = self.mx
        pm = mx.array(p[None, :, :])
        if self.model.prefers_logits_self_conditioning:
            # quantized embeddings take logits: log p is the processed logits up to a constant per position
            return self.model.diffusion_self_conditioning(mx.log(mx.maximum(pm, 1e-30)), None)
        w = self.model.diffusion_prepare_self_conditioning()
        return self.model.diffusion_self_conditioning(mx.log(mx.maximum(pm, 1e-30)), w)
