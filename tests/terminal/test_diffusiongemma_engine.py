"""DiffusionGemma sidecar engine (tools/diffusiongemma): the sampler on fixed logits, and the engine protocol with the
fake model.  No GPU, no Mac, no downloads.  MlxModel is not tested here."""
import os
import queue
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
DG = ROOT / "tools" / "diffusiongemma"
sys.path.insert(0, str(DG))
import sampler as S  # noqa: E402
from model import FakeModel  # noqa: E402


# --- sampler ------------------------------------------------------------------------------------------------------

def test_tau_schedule():
    assert S.tau_at(1, 48) == pytest.approx(0.8)
    assert S.tau_at(25, 48) == pytest.approx(0.4 * (1 - 24 / 48) + 0.4)
    assert S.tau_at(48, 48) == pytest.approx(0.4 * (1 / 48) + 0.4)
    assert S.tau_at(1, 48, temperature=0.4) == pytest.approx(0.4)      # temperature scales both ends
    assert S.tau_at(48, 48, temperature=1.6) == pytest.approx(2 * (0.4 / 48 + 0.4))


def test_accept_count_entropy_budget():
    e = np.array([0.0, 0.03, 0.03, 0.03, 0.5])
    # first m-1 entropies must add to <= 0.1: m=4 -> 0.0+0.03+0.03 = 0.06; m=5 -> 0.09 <= 0.1; m=6 would need 0.59
    assert S.accept_count(e, 0.1) == 5
    assert S.accept_count(np.array([0.2, 0.3]), 0.1) == 1            # the first position is always accepted
    assert S.accept_count(np.array([0.0, 0.1, 0.1]), 0.1) == 3       # sum of the first m-1 may equal the bound
    assert S.accept_count(np.zeros(4), 0.1) == 4                     # never more than C


def _fixed_decode(logits):
    calls = []

    def decode(x, z):
        calls.append((x.copy(), z))
        return logits
    return decode, calls


def test_accept_order_and_renoise():
    # C=4, V=1000.  Position entropies ascending: 2 < 0 < 3 < 1 (ties would go by index).
    V = 1000
    logits = np.zeros((4, V), dtype=np.float32)
    for pos, peak in ((2, 60.0), (0, 14.0), (3, 6.0), (1, 0.0)):
        logits[pos, pos] = peak
    # peak 60 -> entropy 0, 14 -> about 4e-5, 6 -> entropy about 0.4+, 0 -> uniform (about ln 1000)
    decode, calls = _fixed_decode(logits)
    zs = []
    steps_x = []
    rng = np.random.default_rng(3)
    S.denoise_canvas(decode, lambda p: zs.append(p) or "z", rng, V, canvas=4, max_steps=2,
                     entropy_bound=0.1, on_step=lambda n, x0: steps_x.append(x0))
    assert len(calls) == 2 and calls[0][1] is None and calls[1][1] == "z"
    p = S.softmax(logits / S.tau_at(1, 2))
    e = S.entropy(p)
    assert list(np.lexsort((np.arange(4), e))) == [2, 0, 3, 1]
    assert S.accept_count(np.sort(e), 0.1) == 3                      # 2, 0, 3 kept: sum before the last = ~0 <= 0.1
    x2 = calls[1][0]
    # the kept positions 0, 2 hold the token sampled from a peak (60, 14): that is the peak itself
    assert x2[0] == 0 and x2[2] == 2
    assert zs[0].shape == (4, V)


def test_renoise_is_uniform_over_vocab_for_the_rest():
    V = 50
    logits = np.zeros((8, V), dtype=np.float32)       # flat: entropy ln 50 each, so only 1 position is kept
    decode, calls = _fixed_decode(logits)
    seen = set()
    for seed in range(40):
        S.denoise_canvas(decode, lambda p: None, np.random.default_rng(seed), V, canvas=8, max_steps=2)
        seen.update(int(t) for t in calls[-1][0])
    assert len(seen) > 40                              # renoised positions cover the vocabulary


def test_run_to_n_returns_argmax():
    V = 20
    logits = np.random.default_rng(0).normal(size=(6, V)).astype(np.float32)
    decode, calls = _fixed_decode(logits)
    r = S.denoise_canvas(decode, lambda p: None, np.random.default_rng(1), V, canvas=6, max_steps=5)
    assert r.steps == 5 and not r.stopped_early and len(calls) == 5
    assert list(r.tokens) == list(logits.argmax(-1))


def test_adaptive_stop_needs_both_and_never_at_step_one():
    V = 20
    sharp = np.zeros((6, V), dtype=np.float32)
    sharp[np.arange(6), np.arange(6)] = 50.0           # mean entropy ~0 at once
    decode, calls = _fixed_decode(sharp)
    r = S.denoise_canvas(decode, lambda p: None, np.random.default_rng(1), V, canvas=6, max_steps=10)
    assert r.steps == 2 and r.stopped_early            # not step 1, although the entropy is already low

    # low entropy but the argmax changes between steps: no stop
    seq = [sharp, np.roll(sharp, 1, axis=0), np.roll(sharp, 2, axis=0), np.roll(sharp, 3, axis=0)]
    it = iter(seq)
    r = S.denoise_canvas(lambda x, z: next(it), lambda p: None, np.random.default_rng(1), V, canvas=6, max_steps=4)
    assert r.steps == 4 and not r.stopped_early

    # same argmax but high entropy: no stop
    flat = np.zeros((6, V), dtype=np.float32)
    flat[np.arange(6), np.arange(6)] = 0.5
    r = S.denoise_canvas(lambda x, z: flat, lambda p: None, np.random.default_rng(1), V, canvas=6, max_steps=4)
    assert r.steps == 4 and not r.stopped_early


def test_ar_sampler():
    rng = np.random.default_rng(0)
    l = np.array([0.0, 5.0, 1.0, 4.0])
    assert S.sample_token(l, rng) == 1                                        # temperature 0: argmax
    assert all(S.sample_token(l, rng, 1.0, top_k=1) == 1 for _ in range(20))
    assert all(S.sample_token(l, rng, 1.0, top_p=0.5) == 1 for _ in range(20))
    assert S.sample_token(l, rng, 0.0, history=[1], penalty_present=10.0) == 3


def test_fake_model_converges_in_few_steps():
    m = FakeModel()
    m.encode([5, 6, 7], "prompt")
    r = S.denoise_canvas(m.decode, m.self_cond, np.random.default_rng(0), m.vocab_size, canvas=16)
    assert r.stopped_early and r.steps < 12
    assert list(r.tokens) == [m.target(3 + j) for j in range(16)]


# --- protocol -----------------------------------------------------------------------------------------------------

class Proc:
    def __init__(self, *args, env=None):
        e = dict(os.environ, **(env or {}))
        self.p = subprocess.Popen([sys.executable, str(DG / "engine.py"), "--serve", "--model", "fake", *args],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                  bufsize=1, env=e)
        self.q = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()
        self.head = []
        while True:
            line = self.read()
            self.head.append(line)
            if line.startswith("READY"):
                break

    def _pump(self):
        for line in self.p.stdout:
            self.q.put(line.rstrip("\n"))
        self.q.put(None)

    def read(self, timeout=20):
        line = self.q.get(timeout=timeout)
        assert line is not None, "engine ended: " + self.p.stderr.read()
        return line

    def send(self, line):
        self.p.stdin.write(line + "\n")
        self.p.stdin.flush()

    def request(self, line):
        self.send(line)
        lines = []
        while True:
            l = self.read()
            lines.append(l)
            if l.startswith(("DONE", "ERR")):
                return lines

    def close(self):
        try:
            self.send("QUIT")
            self.p.wait(10)
        except Exception:
            self.p.kill()


@pytest.fixture
def eng():
    procs = []

    def start(*args, env=None):
        pr = Proc(*args, env=env)
        procs.append(pr)
        return pr
    yield start
    for pr in procs:
        pr.close()


def toks(lines):
    return [int(l[2:]) for l in lines if l.startswith("T ")]


def done_fields(lines):
    f = lines[-1].split()
    assert f[0] == "DONE"
    return f, dict(t.split("=") for t in f[16:])


def expected_reply(prompt, n):
    m = FakeModel()
    m.encode(prompt, "prompt")
    return [m.target(len(prompt) + i) for i in range(n)]


def test_ready_and_info(eng):
    p = eng("--max-context", "1024", "--canvas", "16")
    assert p.head[-1] == "READY 1024 stop"
    assert any(l.startswith("INFO ") and "diffusion_ignores=" in l for l in p.head)


def test_diffusion_request_single_canvas(eng):
    p = eng("--canvas", "16", "--max-context", "4096")
    prompt = [9, 8, 7, 6]
    lines = p.request("GEN 100 temperature=0.8 seed=5 " + ",".join(map(str, prompt)))
    t = toks(lines)
    m = FakeModel(); m.encode(prompt, "prompt")
    n = m._rlen()
    assert n + 1 <= 16
    assert t == expected_reply(prompt, n) + [1]
    f, kv = done_fields(lines)
    assert int(f[1]) == len(t) and int(f[2]) == len(prompt) and f[5] == "stop"
    assert int(kv["canvases"]) >= 1 and int(kv["steps"]) >= 1
    assert float(kv["tpf"]) == pytest.approx(len(t) / int(kv["steps"]), abs=0.01)
    assert any(l.startswith("PP ") for l in lines)


def test_multi_canvas_and_max_new_cap(eng):
    p = eng("--canvas", "8", "--max-context", "4096")
    prompt = [3, 3, 3]
    lines = p.request("GEN 11 seed=1 " + ",".join(map(str, prompt)))
    f, kv = done_fields(lines)
    assert f[5] == "length" and int(f[1]) == 11 and len(toks(lines)) == 11
    assert toks(lines) == expected_reply(prompt, 11)
    assert int(kv["canvases"]) == 2
    # a short cap lands in the first canvas
    lines = p.request("GEN 3 seed=1 " + ",".join(map(str, prompt)))
    f, kv = done_fields(lines)
    assert f[5] == "length" and int(f[1]) == 3 and int(kv["canvases"]) == 1


def test_stop_ids_from_key_and_default(eng):
    p = eng("--canvas", "16", "--stop-ids", "7")
    prompt = [9, 8, 7, 6]
    ref = expected_reply(prompt, 16)
    lines = p.request("GEN 100 seed=2 " + ",".join(map(str, prompt)))
    t = toks(lines)
    assert t[-1] in (1, 7) and all(x not in (1, 7) for x in t[:-1])
    f, _ = done_fields(lines)
    assert f[5] == "stop" and int(f[1]) == len(t)
    assert ref[:len(t) - 1] == t[:-1]


def test_prefix_reuse_sends_resume(eng):
    p = eng("--canvas", "16")
    a = [5, 6, 7, 8, 9]
    first = p.request("GEN 4 seed=1 " + ",".join(map(str, a)))
    assert not any(l.startswith("RESUME") for l in first)
    second = p.request("GEN 4 seed=1 " + ",".join(map(str, a + [10, 11])))
    assert "RESUME 5" in second
    f, _ = done_fields(second)
    assert int(f[8]) == 5 and int(f[14]) == 2                 # reused, prompt_read
    assert toks(second) == expected_reply(a + [10, 11], 4)
    # the same prompt again still reads one token
    third = p.request("GEN 4 seed=1 " + ",".join(map(str, a + [10, 11])))
    assert "RESUME 6" in third


def test_ar_mode(eng):
    p = eng("--mode", "ar")
    prompt = [4, 4, 4]
    lines = p.request("GEN 5 " + ",".join(map(str, prompt)))
    assert toks(lines) == expected_reply(prompt, 5)
    f, kv = done_fields(lines)
    assert f[5] == "length" and kv["canvases"] == "0" and float(kv["tpf"]) == 1.0
    # per request override, run to the end token
    p2 = eng("--canvas", "16")
    lines = p2.request("GEN 500 mode=ar " + ",".join(map(str, prompt)))
    t = toks(lines)
    f, kv = done_fields(lines)
    assert f[5] == "stop" and t[-1] == 1 and kv["canvases"] == "0"


def test_drafts_only_when_asked_and_throttled(eng):
    p = eng("--canvas", "16", env={"STRATA_FAKE_DELAY_MS": "60"})
    prompt = [9, 8, 7]
    plain = p.request("GEN 100 seed=1 " + ",".join(map(str, prompt)))
    assert not any(l.startswith("DRAFT") for l in plain)
    t0 = time.monotonic()
    lines = p.request("GEN 100 seed=1 drafts=1 " + ",".join(map(str, prompt)))
    dt = time.monotonic() - t0
    d = [l for l in lines if l.startswith("DRAFT ")]
    assert d and len(d[0].split()[1].split(",")) == 16
    assert len(d) <= dt * 10 + 1
    assert lines[-1].startswith("DONE")


def test_stop_cancels(eng):
    p = eng("--canvas", "16", env={"STRATA_FAKE_DELAY_MS": "200"})
    p.send("GEN 1000 seed=1 1,2,3")
    time.sleep(0.5)
    p.send("STOP")
    lines = []
    while True:
        l = p.read()
        lines.append(l)
        if l.startswith(("DONE", "ERR")):
            break
    f, _ = done_fields(lines)
    assert f[5] == "cancel"
    # the next request works
    nxt = p.request("GEN 2 seed=1 1,2,3")
    assert done_fields(nxt)[0][5] == "length"


def test_error_and_quit(eng):
    p = eng("--max-context", "10", "--canvas", "4")
    lines = p.request("GEN 5 " + ",".join(["2"] * 20))
    assert lines[-1].startswith("ERR ")
    p.send("QUIT")
    assert p.p.wait(10) == 0


def test_strata_engine_parses_the_output(tmp_path):
    """serve.server.StrataEngine's own reader accepts READY, T, RESUME, PP, DONE (with the extra keys) and DRAFT."""
    try:
        from serve import server
    except Exception as e:                  # pragma: no cover
        pytest.skip(f"serve.server does not import here: {e}")
    exe = tmp_path / "strata-dgemma"
    exe.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{DG / "engine.py"}" "$@"\n')
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    eng = server.StrataEngine(str(exe), ["--model", "fake", "--canvas", "16", "--max-context", "2048"])
    try:
        assert eng.max_context == 2048 and eng.can_stop
        prompt = [9, 8, 7, 6]
        cancel = threading.Event()
        got = [t for t in eng.generate(prompt, 100, {"temperature": 0.8, "seed": 4}, cancel) if t is not None]
        assert got[-1] == 1 and got[:-1] == expected_reply(prompt, len(got) - 1)
        assert eng.last["finish"] == "stop" and eng.last["generated"] == len(got)
        assert eng.last["prompt_tokens"] == 4
        got2 = [t for t in eng.generate(prompt + [5], 3, {"seed": 4}, cancel) if t is not None]
        assert eng.reused == 4 and len(got2) == 3
    finally:
        eng.close()
