"""DiffusionGemma sidecar engine.  Speaks Strata's engine protocol (serve/server.py StrataEngine) on stdin/stdout.

    python engine.py --serve --model <mlx folder | fake> [--max-context N] [--mode diffusion|ar]
                     [--stop-ids 1,106] [--canvas 256]

Lines out:   INFO k=v ...  |  READY <ctx> stop  |  RESUME n  |  PP done total  |  T <id>  |  DRAFT <ids>  |  DONE ...  |  ERR msg
Lines in:    GEN <max_new> [k=v ...] <id,id,...>  |  STOP  |  QUIT

DONE is `DONE generated prompt_tokens prompt_ms decode_ms finish 0 0 reused 0 0 0 0 0.0 prompt_read 0` and then
steps= canvases= tpf=.  tpf (tokens per forward) is generated tokens / denoising steps; in one-token mode it is 1.0.
One request at a time.  A diffusion request reads the prompt, then repeats: denoise a canvas, commit the tokens up to
and including the first end token, encode the canvas into the cache.
"""
from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sampler  # noqa: E402

IGNORED_IN_DIFFUSION = ("top_k", "top_p", "min_p", "penalty_repeat", "penalty_freq", "penalty_present", "penalty_last_n")
DRAFT_INTERVAL_S = 0.1          # at most 10 DRAFT lines a second
PP_CHUNK = 512


def out(line: str) -> None:
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def parse_gen(line: str):
    """`GEN <max_new> [k=v ...] <ids>` -> (max_new, keys, ids)."""
    parts = line.split()
    max_new = int(parts[1])
    keys, ids = {}, []
    for tok in parts[2:]:
        if "=" in tok:
            k, _, v = tok.partition("=")
            keys[k] = v
        else:
            ids = [int(x) for x in tok.split(",") if x]
    return max_new, keys, ids


def fnum(keys, name, default, cast=float):
    try:
        return cast(keys[name]) if name in keys else default
    except ValueError:
        raise ValueError(f"GEN: bad value for {name}: {keys[name]}")


class Engine:
    def __init__(self, model, max_context, mode, stop_ids, canvas):
        self.model, self.ctx, self.mode, self.stop_ids, self.canvas = model, max_context, mode, set(stop_ids), canvas
        self.cache_ids: list = []        # what the model's cache holds
        self.stop = threading.Event()

    # -- prompt -------------------------------------------------------------------------------------------------
    def read_prompt(self, ids):
        """Reuse the longest common prefix of the cache (at least one token is read again), then encode the rest."""
        m = self.model
        common = 0
        for a, b in zip(self.cache_ids, ids):
            if a != b:
                break
            common += 1
        common = min(common, len(ids) - 1)
        if common > 0:
            try:
                m.truncate(common)
            except Exception:
                m.reset()
                common = 0
        else:
            m.reset()
        self.cache_ids = list(ids[:common])
        if common > 0:
            out(f"RESUME {common}")
        pos = common
        cancelled = False
        while pos < len(ids):
            if self.stop.is_set():
                cancelled = True
                break
            chunk = ids[pos:pos + PP_CHUNK]
            m.encode(chunk, "prompt")
            self.cache_ids.extend(chunk)
            pos += len(chunk)
            out(f"PP {pos} {len(ids)}")
        return common, pos - common, cancelled

    # -- one request --------------------------------------------------------------------------------------------
    def generate(self, max_new, keys, ids):
        t0 = time.perf_counter()
        if not ids:
            raise ValueError("GEN: no prompt tokens")
        if len(ids) >= self.ctx:
            raise ValueError(f"the prompt ({len(ids)} tokens) does not fit the context ({self.ctx})")
        mode = keys.get("mode", self.mode)
        if mode not in ("diffusion", "ar"):
            raise ValueError(f"GEN: unknown mode {mode}")
        stop_ids = set(self.stop_ids)
        if "stop_ids" in keys:
            stop_ids = {int(x) for x in keys["stop_ids"].split(",") if x}
        seed = fnum(keys, "seed", 0, int)
        rng = np.random.default_rng(seed if seed > 0 else None)
        drafts = keys.get("drafts") == "1"
        reused, read, cancelled = self.read_prompt(ids)
        prompt_ms = (time.perf_counter() - t0) * 1000.0
        d0 = time.perf_counter()
        gen = steps = canvases = 0
        finish = "cancel" if cancelled else "length"

        def emit(tokens):
            out("".join(f"T {int(t)}\n" for t in tokens).rstrip("\n"))

        if cancelled or max_new <= 0:
            pass
        elif mode == "ar":
            gen, steps, finish = self.run_ar(max_new, keys, rng, stop_ids, emit)
        else:
            gen, steps, canvases, finish = self.run_diffusion(max_new, keys, rng, stop_ids, emit, drafts)
        decode_ms = (time.perf_counter() - d0) * 1000.0
        tpf = gen / steps if steps else 0.0
        out(f"DONE {gen} {len(ids)} {prompt_ms:.1f} {decode_ms:.1f} {finish} 0 0 {reused} 0 0 0 0 0.0 {read} 0 "
            f"steps={steps} canvases={canvases} tpf={tpf:.2f}")

    def run_ar(self, max_new, keys, rng, stop_ids, emit):
        m = self.model
        temp = fnum(keys, "temperature", 0.0)
        top_k, top_p, min_p = fnum(keys, "top_k", 0, int), fnum(keys, "top_p", 1.0), fnum(keys, "min_p", 0.0)
        pr, pf, pp = fnum(keys, "penalty_repeat", 1.0), fnum(keys, "penalty_freq", 0.0), fnum(keys, "penalty_present", 0.0)
        last_n = fnum(keys, "penalty_last_n", 64, int)
        hist: list = []
        gen = 0
        while gen < max_new:
            if self.stop.is_set():
                return gen, gen, "cancel"
            if m.length + 1 > self.ctx:
                return gen, gen, "length"
            logits = m.last_logits()
            t = sampler.sample_token(logits, rng, temp, top_k, top_p, min_p, hist[-last_n:] if last_n > 0 else None, pr, pf, pp)
            gen += 1
            hist.append(t)
            emit([t])
            if t in stop_ids:
                return gen, gen, "stop"
            m.encode([t], "token")
            self.cache_ids.append(t)
        return gen, gen, "length"

    def run_diffusion(self, max_new, keys, rng, stop_ids, emit, drafts):
        m = self.model
        temperature = fnum(keys, "temperature", sampler.DEFAULT_TEMPERATURE)
        n_steps = fnum(keys, "max_denoising_steps", sampler.MAX_STEPS, int)
        bound = fnum(keys, "entropy_bound", sampler.ENTROPY_BOUND)
        e_stop = fnum(keys, "stop_entropy", sampler.STOP_ENTROPY)
        gen = steps = canvases = 0
        last_draft = [0.0]

        def on_step(n, x0):
            if self.stop.is_set():
                raise _Cancel()
            if drafts:
                now = time.monotonic()
                if now - last_draft[0] >= DRAFT_INTERVAL_S:
                    last_draft[0] = now
                    out("DRAFT " + ",".join(str(int(t)) for t in x0))

        while gen < max_new:
            if self.stop.is_set():
                return gen, steps, canvases, "cancel"
            if m.length + self.canvas > self.ctx:
                return gen, steps, canvases, "length"
            canvases += 1
            try:
                r = sampler.denoise_canvas(m.decode, m.self_cond, rng, m.vocab_size, self.canvas, n_steps, bound,
                                           e_stop, temperature, on_step, step=getattr(m, "step", None))
            except _Cancel:
                return gen, steps, canvases, "cancel"
            steps += r.steps
            toks = [int(t) for t in r.tokens]
            cut = len(toks)
            finish = None
            for i, t in enumerate(toks):
                if t in stop_ids:
                    cut, finish = i + 1, "stop"
                    break
            room = max_new - gen
            if cut > room:
                cut, finish = room, "length"
            elif finish is None and gen + cut >= max_new:
                finish = "length"
            emit(toks[:cut])
            gen += cut
            if finish:
                # the cache holds the prompt and whole canvases; this canvas is not added (the next prompt reads it)
                return gen, steps, canvases, finish
            m.encode(toks, "canvas")
            self.cache_ids.extend(toks)
            if hasattr(m, "clear_cache"):
                m.clear_cache()
        return gen, steps, canvases, "length"


class _Cancel(Exception):
    pass


def load_model(spec: str, vocab: int):
    import model as M
    if spec == "fake":
        return M.FakeModel(vocab=vocab)
    return M.MlxModel(spec)


def reader(q: "queue.Queue", engine: Engine) -> None:
    """stdin -> queue.  STOP sets the flag here, in order with the GEN lines (a GEN clears it before it is queued)."""
    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        head = line.split(None, 1)[0]
        if head == "STOP":
            engine.stop.set()
        elif head == "GEN":
            engine.stop.clear()
            q.put(line)
        elif head == "QUIT":
            engine.stop.set()
            q.put(line)
            return
        else:
            q.put(line)
    q.put("QUIT")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-context", type=int, default=32768)
    ap.add_argument("--mode", choices=("diffusion", "ar"), default="diffusion")
    ap.add_argument("--stop-ids", default="")
    ap.add_argument("--eos", type=int, default=1)
    ap.add_argument("--canvas", type=int, default=sampler.CANVAS)
    ap.add_argument("--vocab", type=int, default=64, help="vocabulary of the fake model")
    a, _unknown = ap.parse_known_args(argv)
    if not a.serve:
        ap.error("only --serve is supported")
    stop_ids = {a.eos} | {int(x) for x in a.stop_ids.split(",") if x}
    model = load_model(a.model, a.vocab)
    eng = Engine(model, a.max_context, a.mode, stop_ids, a.canvas)
    out(f"INFO arch=diffusion_gemma mode={a.mode} canvas={a.canvas} vocab={model.vocab_size}")
    out("INFO diffusion_ignores=" + ",".join(IGNORED_IN_DIFFUSION))
    out(f"READY {a.max_context} stop")
    q: "queue.Queue" = queue.Queue()
    threading.Thread(target=reader, args=(q, eng), daemon=True).start()
    while True:
        line = q.get()
        head = line.split(None, 1)[0]
        if head == "QUIT":
            return 0
        if head != "GEN":
            out(f"ERR unknown command {head}")
            continue
        try:
            max_new, keys, ids = parse_gen(line)
            eng.generate(max_new, keys, ids)
        except Exception as e:      # the request fails, the engine stays up; the cache may be half-written
            eng.cache_ids = []
            try:
                eng.model.reset()
            except Exception:
                pass
            out(f"ERR {type(e).__name__}: {e}".replace("\n", " "))


if __name__ == "__main__":
    sys.exit(main())
