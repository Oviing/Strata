"""DiffusionGemma through serve/server.py: `--engine strata` with the real sidecar protocol (a scripted stand-in and
the sidecar with its fake model), a tokenizer.json built here, and the real Gemma 4 chat template.  No GPU, no Mac."""
import json
import os
import socket
import stat
import subprocess
import sys
import textwrap
import time
import urllib.request
from pathlib import Path

import pytest

tokenizers = pytest.importorskip("tokenizers")
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "serve"))
from serve import gemma4, hf_tokenizer  # noqa: E402
from serve.server import ChatTemplate  # noqa: E402

SPECIALS = ["<pad>", "<eos>", "<bos>", "<unk>"]
MARKERS = ["<|turn>", "<turn|>", "<|channel>", "<channel|>", "<|tool_call>", "<tool_call|>", "<|tool>", "<tool|>",
           "<|tool_response>", "<tool_response|>", "<|think|>", '<|"|>']
TURN_END = 106


def make_tokenizer(folder: Path) -> None:
    """A character-level tokenizer.json: ids 0-3 pad/eos/bos/unk, then the characters, filler up to 105, <turn|> at 106."""
    chars = ["▁"] + [chr(c) for c in range(33, 127)] + ["\n"]
    vocab = {s: i for i, s in enumerate(SPECIALS)}
    for c in chars:
        vocab[c] = len(vocab)
    while len(vocab) < TURN_END - 1:
        vocab[f"▁f{len(vocab)}"] = len(vocab)
    ids = {"<|turn>": TURN_END - 1, "<turn|>": TURN_END}
    nxt = TURN_END + 1
    for m in MARKERS[2:]:
        ids[m] = nxt
        nxt += 1
    vocab["<|turn>"] = ids["<|turn>"]
    for m in MARKERS[1:]:
        vocab[m] = ids[m]
    tk = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tk.normalizer = normalizers.Replace(" ", "▁")
    tk.pre_tokenizer = pre_tokenizers.Split(tokenizers.Regex(r"[\s\S]"), "isolated")
    tk.add_special_tokens([tokenizers.AddedToken(s, special=True) for s in SPECIALS + MARKERS])
    folder.mkdir(parents=True, exist_ok=True)
    tk.save(str(folder / "tokenizer.json"))
    (folder / "generation_config.json").write_text(json.dumps({"bos_token_id": 2, "eos_token_id": [1, 106, 50]}))


@pytest.fixture(scope="module")
def tokdir(tmp_path_factory):
    d = tmp_path_factory.mktemp("tok")
    make_tokenizer(d)
    return d


@pytest.fixture(scope="module")
def tok(tokdir):
    return hf_tokenizer.load(tokdir)


# --- the real template -----------------------------------------------------------------------------------------------

TOOLS = [{"name": "get_weather", "description": "Weather of a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}]


def test_template_plain_and_thinking():
    t = ChatTemplate(gemma4.FIXTURE_TEMPLATE)
    off = t.render([{"role": "user", "content": "hi"}], tools=None, enable_thinking=False, bos_token="<bos>")
    on = t.render([{"role": "user", "content": "hi"}], tools=None, enable_thinking=True, bos_token="<bos>")
    assert "<|think|>" not in off and "<|think|>" in on
    assert on.startswith("<bos><|turn>system\n<|think|>") and off.endswith("<|turn>user\nhi<turn|>\n<|turn>model\n")
    assert "<|turn>user\nhi<turn|>" in on


def test_template_tools_and_multiturn_drops_old_thoughts():
    t = ChatTemplate(gemma4.FIXTURE_TEMPLATE)
    msgs = [{"role": "user", "content": "weather in Zagreb?"},
            {"role": "assistant", "content": "", "reasoning_content": "OLD-THOUGHT",
             "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "Zagreb"}}}]},
            {"role": "tool", "tool_call_id": "c1", "name": "get_weather", "content": "sunny"},
            {"role": "assistant", "content": "It is sunny."},
            {"role": "user", "content": "thanks"}]
    out = t.render(msgs, tools=[{"type": "function", "function": TOOLS[0]}], enable_thinking=True, bos_token="<bos>")
    assert "<|tool>" in out and "get_weather" in out and "<tool|>" in out
    assert "<|tool_call>call:get_weather{" in out and "<tool_call|>" in out
    assert "<|tool_response>" in out and "sunny" in out and "<tool_response|>" in out
    assert "OLD-THOUGHT" not in out                      # earlier thoughts are dropped from the history
    assert out.endswith("<|turn>user\nthanks<turn|>\n<|turn>model\n")


def test_pick_template():
    p, swapped = gemma4.pick_template(None)
    assert p == gemma4.FIXTURE_TEMPLATE and not swapped


# --- the server with a sidecar -----------------------------------------------------------------------------------------

FAKE_SIDECAR = textwrap.dedent('''
    import json, os, sys
    work = os.environ["FAKE_WORK"]
    def out(s): sys.stdout.write(s + "\\n"); sys.stdout.flush()
    out("INFO arch=diffusion_gemma mode=diffusion canvas=256 vocab=2000")
    out("READY 8192 stop")
    for raw in sys.stdin:
        line = raw.strip()
        if line == "QUIT": break
        if not line.startswith("GEN"): continue
        with open(os.path.join(work, "gen.log"), "a") as f: f.write(line + "\\n")
        q = os.path.join(work, "replies.json")
        replies = json.load(open(q))
        ids = replies.pop(0)
        json.dump(replies, open(q, "w"))
        head = line.split()
        out("PP 5 5")
        if "drafts=1" in head:
            half = ids[: max(1, len(ids) // 2)]
            out("DRAFT " + ",".join(map(str, half)))
            out("DRAFT " + ",".join(map(str, ids)))
        for t in ids:
            out(f"T {t}")
        stopped = bool(ids) and ids[-1] in (1, 106)
        out(f"DONE {len(ids)} 5 1.0 2.0 {'stop' if stopped else 'length'} 0 0 0 0 0 0 0 0.0 5 0 steps=7 canvases=1 tpf={len(ids)/7:.2f}")
''')


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Srv:
    def __init__(self, work: Path, tokdir: Path, scripted: bool):
        self.work, self.port = work, free_port()
        work.mkdir(parents=True, exist_ok=True)
        (work / "replies.json").write_text("[]")
        if scripted:
            (work / "fake.py").write_text(FAKE_SIDECAR)
            body = f'exec {sys.executable} {work / "fake.py"} "$@"'
        else:
            body = f'exec {sys.executable} {ROOT / "tools/diffusiongemma/engine.py"} "$@"'
        exe = work / "launcher.sh"
        exe.write_text("#!/bin/sh\n" + body + "\n")
        exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
        args = [] if scripted else ["--model", "fake", "--vocab", "100", "--stop-ids", "1,106", "--canvas", "16"]
        cfg = {"exe": str(exe), "args": args, "cwd": str(work), "log": str(work / "engine.log"), "arch": "diffusion_gemma",
               "tokenizer": str(tokdir), "model_name": "diffusiongemma-26b-a4b-it"}
        (work / "cfg.json").write_text(json.dumps(cfg))
        env = {**os.environ, "FAKE_WORK": str(work), "PYTHONPATH": str(ROOT / "serve") + os.pathsep + str(ROOT / "tools")}
        self.log = open(work / "server.log", "w")
        self.proc = subprocess.Popen([sys.executable, str(ROOT / "serve/server.py"), "--engine", "strata", "--config",
                                      str(work / "cfg.json"), "--port", str(self.port), "--host", "127.0.0.1"],
                                     stdout=self.log, stderr=subprocess.STDOUT, env=env, cwd=str(ROOT))
        deadline = time.time() + 60
        while time.time() < deadline:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2).read()
                return
            except Exception:
                if self.proc.poll() is not None:
                    break
                time.sleep(0.3)
        self.close()
        raise RuntimeError("server did not start:\n" + (work / "server.log").read_text())

    def reply(self, *texts_or_ids, tok):
        items = [tok.encode(t, parse_special=True) if isinstance(t, str) else list(t) for t in texts_or_ids]
        (self.work / "replies.json").write_text(json.dumps(items))

    def post(self, path, body, stream=False):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        raw = urllib.request.urlopen(req, timeout=60).read().decode()
        if not stream:
            return json.loads(raw)
        return [json.loads(l[6:]) for l in raw.splitlines() if l.startswith("data: ") and l != "data: [DONE]"]

    def gens(self):
        p = self.work / "gen.log"
        return p.read_text().splitlines() if p.exists() else []

    def close(self):
        self.proc.terminate()
        try:
            self.proc.wait(10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.log.close()


@pytest.fixture(scope="module")
def srv(tmp_path_factory, tokdir):
    s = Srv(tmp_path_factory.mktemp("scripted"), tokdir, True)
    yield s
    s.close()


def merged(chunks):
    reasoning = "".join(c["choices"][0]["delta"].get("reasoning_content") or "" for c in chunks if c.get("choices"))
    content = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks if c.get("choices"))
    return reasoning, content


def test_streaming_thinking_on(srv, tok):
    srv.reply("<|channel>thought\nlet me think<channel|>The answer.<turn|>", tok=tok)
    chunks = srv.post("/v1/chat/completions", {"model": "diffusiongemma-26b-a4b-it", "stream": True,
                      "messages": [{"role": "user", "content": "hi"}]}, stream=True)
    reasoning, content = merged(chunks)
    assert reasoning == "let me think" and content == "The answer."
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert not any("strata_draft" in c["choices"][0]["delta"] for c in chunks if c.get("choices"))
    prompt_ids = [int(x) for x in srv.gens()[-1].split()[-1].split(",")]
    assert tok.decode(prompt_ids).startswith("<bos><|turn>system\n<|think|>")        # thinking is the server's default
    assert "temperature=0.8" in srv.gens()[-1]


def test_streaming_thinking_off(srv, tok):
    srv.reply("<|channel>thought\n<channel|>Plain.<turn|>", tok=tok)
    chunks = srv.post("/v1/chat/completions", {"stream": True, "reasoning_effort": "none",
                      "messages": [{"role": "user", "content": "hi"}]}, stream=True)
    reasoning, content = merged(chunks)
    assert reasoning == "" and content == "Plain."
    prompt_ids = [int(x) for x in srv.gens()[-1].split()[-1].split(",")]
    assert "<|think|>" not in tok.decode(prompt_ids)


def test_effort_levels_are_on_off_only(srv, tok):
    srv.reply("<|channel>thought\nx<channel|>ok<turn|>", tok=tok)
    srv.post("/v1/chat/completions", {"reasoning_effort": "low", "messages": [{"role": "user", "content": "hi"}]})
    prompt_ids = [int(x) for x in srv.gens()[-1].split()[-1].split(",")]
    assert "<|think|>" in tok.decode(prompt_ids)


def test_tool_call(srv, tok):
    srv.reply('<|channel>thought\n<channel|><|tool_call>call:get_weather{city:<|"|>Zagreb<|"|>}<tool_call|><turn|>', tok=tok)
    r = srv.post("/v1/chat/completions", {"messages": [{"role": "user", "content": "weather?"}], "reasoning_effort": "none",
                 "tools": [{"type": "function", "function": TOOLS[0]}]})
    ch = r["choices"][0]
    assert ch["finish_reason"] == "tool_calls"
    call = ch["message"]["tool_calls"][0]["function"]
    assert call["name"] == "get_weather" and json.loads(call["arguments"]) == {"city": "Zagreb"}


def test_anthropic_messages(srv, tok):
    srv.reply("<|channel>thought\n<channel|>Hello there.<turn|>", tok=tok)
    r = srv.post("/v1/messages", {"model": "diffusiongemma-26b-a4b-it", "max_tokens": 100,
                 "messages": [{"role": "user", "content": "hi"}]})
    assert r["type"] == "message" and r["stop_reason"] == "end_turn"
    assert "".join(b.get("text", "") for b in r["content"] if b["type"] == "text") == "Hello there."


def test_drafts_only_when_asked(srv, tok):
    srv.reply("<|channel>thought\n<channel|>Draft me.<turn|>", "<|channel>thought\n<channel|>Draft me.<turn|>", tok=tok)
    body = {"stream": True, "reasoning_effort": "none", "messages": [{"role": "user", "content": "hi"}]}
    plain = srv.post("/v1/chat/completions", body, stream=True)
    assert "drafts=1" not in srv.gens()[-1]
    assert not any("strata_draft" in c["choices"][0]["delta"] for c in plain if c.get("choices"))
    asked = srv.post("/v1/chat/completions", {**body, "strata": {"drafts": True}}, stream=True)
    assert "drafts=1" in srv.gens()[-1]
    drafts = [c["choices"][0]["delta"]["strata_draft"] for c in asked if c.get("choices") and "strata_draft" in c["choices"][0]["delta"]]
    assert drafts and "Draft me." in drafts[-1]                    # the whole canvas text each time
    assert merged(asked)[1] == "Draft me."
    # a non-streaming reply is unchanged
    srv.reply("<|channel>thought\n<channel|>Draft me.<turn|>", tok=tok)
    r = srv.post("/v1/chat/completions", {**body, "stream": False, "strata": {"drafts": True}})
    assert "strata_draft" not in json.dumps(r)


def test_sampling_keys_forwarded(srv, tok):
    srv.reply("<|channel>thought\n<channel|>k<turn|>", tok=tok)
    srv.post("/v1/chat/completions", {"reasoning_effort": "none", "mode": "ar", "max_denoising_steps": 12, "entropy_bound": 0.2,
             "stop_entropy": 0.01, "seed": 5, "messages": [{"role": "user", "content": "hi"}]})
    g = srv.gens()[-1]
    for k in ("mode=ar", "max_denoising_steps=12", "entropy_bound=0.2", "stop_entropy=0.01", "seed=5"):
        assert k in g.split()
    srv.reply("<|channel>thought\n<channel|>k<turn|>", tok=tok)
    srv.post("/v1/chat/completions", {"reasoning_effort": "none", "messages": [{"role": "user", "content": "hi"}]})
    assert "mode=" not in srv.gens()[-1] and "max_denoising_steps" not in srv.gens()[-1]


def test_done_extras_kept(srv, tok):
    srv.reply("<|channel>thought\n<channel|>k<turn|>", tok=tok)
    srv.post("/v1/chat/completions", {"reasoning_effort": "none", "messages": [{"role": "user", "content": "hi"}]})
    from serve.server import StrataEngine
    e = StrataEngine.__new__(StrataEngine)
    e.last = {}
    e._parse_done("DONE 3 5 1.0 2.0 stop 0 0 0 0 0 0 0 0.0 5 0 steps=7 canvases=2 tpf=0.43")
    assert e.last["steps"] == 7 and e.last["canvases"] == 2 and e.last["tpf"] == pytest.approx(0.43)


def test_real_sidecar_fake_model(tmp_path_factory, tokdir):
    s = Srv(tmp_path_factory.mktemp("real"), tokdir, False)
    try:
        body = {"stream": True, "reasoning_effort": "none", "max_tokens": 200, "mode": "diffusion",
                "strata": {"drafts": True}, "messages": [{"role": "user", "content": "hi"}]}
        chunks = s.post("/v1/chat/completions", body, stream=True)
        assert chunks[-1]["choices"][0]["finish_reason"] in ("stop", "length")
        assert any("strata_draft" in c["choices"][0]["delta"] for c in chunks if c.get("choices"))
        r = s.post("/v1/chat/completions", {"reasoning_effort": "none", "mode": "ar", "max_tokens": 20,
                   "messages": [{"role": "user", "content": "hi"}]})
        assert r["usage"]["completion_tokens"] > 0
    finally:
        s.close()
