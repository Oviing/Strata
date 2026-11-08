"""serve/gemma4.py - Gemma 4's (DiffusionGemma's) reasoning and tool-call text for the server.

The model writes (the DiffusionGemma report, appendix D)

    <|channel>thought
    ...its reasoning...<channel|>The answer.
    <|tool_call>call:get_weather{city:<|"|>Zagreb<|"|>,days:3,opts:{metric:true,tags:[<|"|>a<|"|>]}}<tool_call|>

The thought channel is always there; with thinking off it is empty.  A string is written between `<|"|>` marks and is not
escaped (it may hold braces, commas, quotes); keys are bare; every other value is JSON-like.

The server's OutputParser (serve/frontend.py) reads the Qwen form: reasoning up to `</think>` and
`<tool_call><function=NAME><parameter=P>...`.  Gemma4Translator rewrites the Gemma text into that form as it streams
(holding back only a tag that is still arriving), and Gemma4OutputParser is the OutputParser behind it, as
serve/dsml.py's DsmlOutputParser is for DeepSeek.  The reply is read as reasoning from its start, like a Qwen reply after
`<think>`: an empty or missing thought channel gives `</think>` at once, so there is no reasoning and the answer is content.

A tool call is passed on when it is complete (`<tool_call|>`), not while its arguments are being written.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from serve.frontend import Event, OutputParser

QUOTE = '<|"|>'                         # the string mark
CH_OPEN, CH_CLOSE = "<|channel>", "<channel|>"
CALL_OPEN, CALL_CLOSE = "<|tool_call>", "<tool_call|>"
RESP_OPEN = "<|tool_response>"
_TAGS = (CH_OPEN, CH_CLOSE, CALL_OPEN, CALL_CLOSE, RESP_OPEN)
_CHANNEL_NAME_MAX = 40


# ------------------------------------------------------------------------------------------------ arguments
class ArgError(ValueError):
    pass


_NUMBER = re.compile(r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?\Z")
_JSON_ESC = {'"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t"}


class _Args:
    """Reader of Gemma's JSON-like value text."""

    def __init__(self, s: str):
        self.s, self.i = s, 0

    def ws(self) -> None:
        s, n = self.s, len(self.s)
        while self.i < n and s[self.i] in " \t\r\n":
            self.i += 1

    def peek(self) -> str:
        return self.s[self.i] if self.i < len(self.s) else ""

    def string(self) -> str:
        """At a `<|"|>` mark or a `"`: the string, left as written for a mark, unescaped for a plain quote."""
        s = self.s
        if s.startswith(QUOTE, self.i):
            start = self.i + len(QUOTE)
            end = s.find(QUOTE, start)
            if end < 0:
                raise ArgError("a string is not closed")
            self.i = end + len(QUOTE)
            return s[start:end]
        j, out = self.i + 1, []
        while j < len(s):
            c = s[j]
            if c == '"':
                self.i = j + 1
                return "".join(out)
            if c == "\\" and j + 1 < len(s):
                e = s[j + 1]
                if e == "u" and re.fullmatch(r"[0-9a-fA-F]{4}", s[j + 2:j + 6]):
                    out.append(chr(int(s[j + 2:j + 6], 16)))
                    j += 6
                    continue
                out.append(_JSON_ESC.get(e, e))
                j += 2
                continue
            out.append(c)
            j += 1
        raise ArgError("a string is not closed")

    def value(self):
        self.ws()
        c = self.peek()
        if c == "":
            raise ArgError("a value is missing")
        if self.s.startswith(QUOTE, self.i) or c == '"':
            return self.string()
        if c == "{":
            return self.obj()
        if c == "[":
            return self.arr()
        j, s = self.i, self.s
        while j < len(s) and s[j] not in ",}]" :
            j += 1
        tok = s[self.i:j].strip()
        self.i = j
        if tok == "":
            raise ArgError("a value is missing")
        if tok in ("true", "false"):
            return tok == "true"
        if tok in ("null", "none", "None"):
            return None
        if _NUMBER.match(tok):
            return float(tok) if re.search(r"[.eE]", tok) else int(tok)
        return tok                                          # an unquoted word is a string

    def key(self) -> str:
        self.ws()
        if self.s.startswith(QUOTE, self.i) or self.peek() == '"':
            return self.string()
        j = self.s.find(":", self.i)
        if j < 0:
            raise ArgError("a key has no ':'")
        k = self.s[self.i:j].strip()
        if not k or any(ch in k for ch in "{}[],"):
            raise ArgError("bad key %r" % k)
        self.i = j
        return k

    def obj(self) -> dict:
        self.i += 1                                         # '{'
        out: dict = {}
        while True:
            self.ws()
            if self.peek() == "}":
                self.i += 1
                return out
            k = self.key()
            self.ws()
            if self.peek() != ":":
                raise ArgError("expected ':' after %r" % k)
            self.i += 1
            out[k] = self.value()
            self.ws()
            c = self.peek()
            if c == ",":
                self.i += 1
            elif c != "}":
                raise ArgError("expected ',' or '}'")

    def arr(self) -> list:
        self.i += 1                                         # '['
        out: list = []
        while True:
            self.ws()
            if self.peek() == "]":
                self.i += 1
                return out
            out.append(self.value())
            self.ws()
            c = self.peek()
            if c == ",":
                self.i += 1
            elif c != "]":
                raise ArgError("expected ',' or ']'")


def parse_args(text: str) -> dict:
    """`{city:<|"|>Zagreb<|"|>,days:3}` -> {"city": "Zagreb", "days": 3}.  ArgError (a ValueError) when it is not an object."""
    r = _Args(text)
    r.ws()
    if r.peek() != "{":
        raise ArgError("the arguments are not an object")
    v = r.obj()
    r.ws()
    if r.i != len(text):
        raise ArgError("text after the arguments")
    return v


_CALL_HEAD = re.compile(r"\s*call\s*:\s*([^\s{]+)\s*")


def parse_call(body: str):
    """The text between `<|tool_call>` and `<tool_call|>` -> (name, arguments dict).  ValueError when it is not a call."""
    m = _CALL_HEAD.match(body)
    if not m:
        raise ArgError("not a call")
    rest = body[m.end():].strip()
    return m.group(1), (parse_args(rest) if rest else {})


def call_end(buf: str) -> int:
    """Where `<tool_call|>` starts in `buf` outside a string, or -1 (the call is not complete)."""
    i, n, in_q, in_dq = 0, len(buf), False, False
    while i < n:
        if not in_dq and buf.startswith(QUOTE, i):
            in_q = not in_q
            i += len(QUOTE)
            continue
        c = buf[i]
        if not in_q:
            if in_dq:
                if c == "\\":
                    i += 2
                    continue
                if c == '"':
                    in_dq = False
            elif c == '"':
                in_dq = True
            elif buf.startswith(CALL_CLOSE, i):
                return i
        i += 1
    return -1


def call_text(name: str, args: dict, on_param=None) -> str:
    """A call in the Qwen form the OutputParser reads."""
    parts = [f"<tool_call>\n<function={name}>\n"]
    for k, v in args.items():
        if isinstance(v, str):
            if on_param is not None:
                on_param(name, k, True)
            val = v
        else:
            val = json.dumps(v, ensure_ascii=False)
        parts.append(f"<parameter={k}>\n{val}\n</parameter>\n")
    parts.append("</function>\n</tool_call>\n")
    return "".join(parts)


# ------------------------------------------------------------------------------------------------ the translator
def _could_be_tag(s: str) -> bool:
    return any(t.startswith(s) for t in _TAGS)


class Gemma4Translator:
    """Gemma 4 text -> the Qwen text (reasoning up to `</think>`, `<tool_call>` blocks), a delta at a time.

    The parser behind it starts in the reasoning state.  `lead` is true until the first thing of the reply is known: a thought
    channel (its text is the reasoning) or anything else (`</think>` first, the reply is the answer)."""

    def __init__(self, on_param=None):
        self.buf = ""
        self.state = "lead"        # lead | thought | text | call
        self.on_param = on_param
        self.after_call = False

    @property
    def held(self) -> str:
        return self.buf

    def feed(self, delta: str) -> str:
        self.buf += delta
        return self._run(final=False)

    def flush(self) -> str:
        return self._run(final=True)

    # --------------------------------------------------------------------------------------------
    def _run(self, final: bool) -> str:
        out: list[str] = []
        while True:
            if self.state == "call":
                if not self._call(out, final):
                    break
                continue
            if self.state == "lead":
                s = self.buf.lstrip()
                if not s:
                    if final:
                        self.buf = ""
                        self.state = "text"
                        out.append("</think>")
                    break
                if CH_OPEN.startswith(s) and s != CH_OPEN and not final:
                    break                                       # may still become the channel tag
                if not s.startswith(CH_OPEN):
                    self.state = "text"
                    out.append("</think>")
                    continue
                self.buf = s
            if not self.buf:
                break
            lt = self.buf.find("<")
            if lt < 0:
                self._text(self.buf, out)
                self.buf = ""
                break
            if lt:
                self._text(self.buf[:lt], out)
                self.buf = self.buf[lt:]
            r = self._tag(out, final)
            if r == "wait":
                if final:
                    self._text(self.buf, out)
                    self.buf = ""
                break
            if r is None:
                self._text("<", out)
                self.buf = self.buf[1:]
        return "".join(out)

    def _text(self, text: str, out: list[str]) -> None:
        if self.after_call and text.strip() == "":
            return                                              # the layout between calls
        if text.strip():
            self.after_call = False
        out.append(text)

    def _tag(self, out: list[str], final: bool):
        s = self.buf
        if s.startswith(CH_OPEN):
            rest = s[len(CH_OPEN):]
            nl = rest.find("\n")
            if nl < 0:
                if not final and len(rest) <= _CHANNEL_NAME_MAX and re.fullmatch(r"[A-Za-z_]*", rest):
                    return "wait"
                name, used = "", len(CH_OPEN)                   # not a channel line: only the tag is dropped
            else:
                name = rest[:nl].strip()
                used = len(CH_OPEN) + nl + 1
                if len(name) > _CHANNEL_NAME_MAX or not re.fullmatch(r"[A-Za-z_]*", name):
                    name, used = "", len(CH_OPEN)
            self.buf = s[used:]
            if name == "thought" and self.state in ("lead", "thought"):
                self.state = "thought"                          # the reasoning text follows
            elif self.state == "lead":
                self.state = "text"
                out.append("</think>")
            return True
        if s.startswith(CH_CLOSE):
            self.buf = s[len(CH_CLOSE):]
            if self.state in ("thought", "lead"):
                out.append("</think>")
            self.state = "text"
            return True
        if s.startswith(CALL_OPEN):
            self.buf = s[len(CALL_OPEN):]
            self.state = "call"
            return True
        if s.startswith(CALL_CLOSE):                            # a stray closer
            self.buf = s[len(CALL_CLOSE):]
            return True
        if s.startswith(RESP_OPEN):
            self.buf = s[len(RESP_OPEN):]
            if not self.after_call:
                out.append(RESP_OPEN)                           # not the stop that follows a call: it is text
            return True
        if len(s) < max(map(len, _TAGS)) and _could_be_tag(s) and not final:
            return "wait"
        return None

    def _call(self, out: list[str], final: bool) -> bool:
        """In a call.  True: go on; False: wait for more text."""
        end = call_end(self.buf)
        if end < 0:
            if not final:
                return False
            body, rest = self.buf, ""
            raw = CALL_OPEN + body
        else:
            body, rest = self.buf[:end], self.buf[end + len(CALL_CLOSE):]
            raw = CALL_OPEN + body + CALL_CLOSE
        try:
            name, args = parse_call(body)
        except ValueError:
            if end < 0:                                         # cut off and unreadable: it is text
                out.append(raw)
                self.buf, self.state = "", "text"
                return False
            out.append(raw)                                     # a closed block that is not a call: text, as written
            self.buf, self.state = rest, "text"
            self.after_call = False
            return True
        out.append(call_text(name, args, self.on_param))
        self.buf, self.state = rest, "text"
        self.after_call = True
        return end >= 0


# ------------------------------------------------------------------------------------------------ the parser
class Gemma4OutputParser(OutputParser):
    """OutputParser for Gemma 4: its thought channel is the reasoning and its calls are read as the template-form calls they are."""

    def __init__(self, thinking: bool = True, *args, **kwargs):
        # The reply always starts in the reasoning state: the translator ends it with `</think>` at once when the thought
        # channel is empty or missing, which is what thinking off looks like.
        super().__init__(True, *args, **kwargs)
        self.thinking = thinking
        self.gemma = Gemma4Translator(self._on_param)
        self._reasoning: list[str] = []
        self._answered = False

    def _watch(self, evs):
        for ev in evs:
            if ev.kind == "reasoning":
                self._reasoning.append(ev.text)
            elif ev.kind in ("tool_start", "tool_call") or (ev.kind == "content" and ev.text.strip()):
                self._answered = True
        return evs

    def _on_param(self, tool, name, is_string) -> None:
        """A string argument is a string whatever it looks like ("123", "true")."""
        spec = self.schemas.get(tool)
        if spec is None:
            return
        params = dict(spec.get("parameters") or {})
        props = dict(params.get("properties") or {})
        if (props.get(name) or {}).get("type") != "string":
            props[name] = {**(props.get(name) or {}), "type": "string"}
            params["properties"] = props
            self.schemas[tool] = {**spec, "parameters": params}
        if getattr(self, "scall", None) is not None and self.scall.name == tool:
            self.sdeclared[name] = "string"

    def feed(self, delta: str):
        return self._watch(super().feed(self.gemma.feed(delta)))

    def finish(self, reason: str | None = None):
        rest = self.gemma.flush()
        out = self._watch(super().feed(rest) if rest else [])
        out += self._watch(super().finish(reason))
        text = "".join(self._reasoning).strip()
        if reason in (None, "stop") and text and not self._answered:
            # the thought channel was never closed and nothing followed: the text is the answer (as serve/dsml.py does)
            self._answered = True
            out.append(Event("content", text))
        return out

    @property
    def clean(self) -> bool:
        return not self.buf and not self.gemma.held


# ------------------------------------------------------------------------------------------------ template, forced call
FIXTURE_TEMPLATE = Path(__file__).resolve().parent / "fixtures" / "diffusiongemma" / "chat_template.jinja"


def gemma4_template_ok(text: str) -> bool:
    """The Gemma 4 template writes turns as `<|turn>role` ... `<turn|>` and calls as `<|tool_call>`."""
    return "<|turn>" in text and "<turn|>" in text


def pick_template(pack_template: Path | None) -> tuple[Path, bool]:
    """(the template file to render with, True when it replaced a pack template of another spelling)."""
    if pack_template is not None and pack_template.exists():
        try:
            if gemma4_template_ok(pack_template.read_text(encoding="utf-8")):
                return pack_template, False
        except OSError:
            return pack_template, False
        return FIXTURE_TEMPLATE, True
    return FIXTURE_TEMPLATE, False


def to_gemma4_force(force: str | None) -> str | None:
    """forced_call's opening of a call in the Qwen form -> the same opening in Gemma's spelling, after the empty thought
    channel the model writes when it does not think."""
    if not force:
        return force
    head = CH_OPEN + "thought\n" + CH_CLOSE + CALL_OPEN + "call:"
    marker = "<function="
    at = force.find(marker)
    name = force[at + len(marker):] if at >= 0 else ""
    if name.endswith(">\n"):
        return head + name[:-2] + "{"
    return head + name
