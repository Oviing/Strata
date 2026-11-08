"""serve/hf_tokenizer.py - a model's own `tokenizer.json` for the server (DiffusionGemma, and Gemma 4 in general).

The other families ship their vocabulary inside the pack and use tools/strata_tokenizer.py.  A Gemma tokenizer is a
SentencePiece-style one (a space is written "▁", a byte that no piece covers is a `<0xHH>` token) kept in the Hugging Face
`tokenizer.json` format.  HFTokenizer reads that file with the `tokenizers` library and gives the server the same
methods strata_tokenizer.Tokenizer has: encode, encode_marked, decode, token_bytes, special_ids, control_tokens,
max_special_len and ALWAYS.

One difference from the library's own encode: `parse_special` is honoured.  The library always turns the text of an
added token into its id.  The server must not do that for text a user wrote, so the text is cut at the added tokens here
and the stretches between them are encoded by a copy of the tokenizer that has no added tokens.  Nothing crosses a
special token, the same rule as strata_tokenizer's (this is what makes PromptEncoder's reuse of an earlier prompt exact).

Added tokens the file marks `special: true` are CONTROL tokens (`<|turn>`, `<turn|>`, `<|channel>`...): ids only when
`parse_special` is set.  Added tokens that are not special (`<h1>`...) are matched always (ALWAYS).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from tokenizers import Tokenizer as _Tokenizer

_BYTE_TOKEN = re.compile(r"^<0x([0-9A-Fa-f]{2})>$")


class HFTokenizer:
    def __init__(self, tokenizer_json: str | Path, special_ids: dict | None = None):
        path = Path(tokenizer_json)
        text = path.read_text(encoding="utf-8")
        self.tk = _Tokenizer.from_str(text)
        data = json.loads(text)
        added = data.get("added_tokens") or []
        # the same tokenizer without added tokens: it encodes the stretches between special tokens
        data["added_tokens"] = []
        self._plain = _Tokenizer.from_str(json.dumps(data))
        self.special_tokens: dict[str, int] = {}                  # text -> id of every added token
        control: list[str] = []
        always: list[str] = []
        for t in added:
            content, tid = t["content"], int(t["id"])
            if not content:
                continue
            self.special_tokens[content] = tid
            (control if t.get("special") else always).append(content)
        self.control_tokens = control
        self.ALWAYS = tuple(always)
        self.max_special_len = max((len(s) for s in self.special_tokens), default=1)
        self._always = set(always)
        ordered = sorted(self.special_tokens, key=len, reverse=True)      # the longest literal wins at a position
        self._special_re = re.compile("|".join(re.escape(s) for s in ordered)) if ordered else None
        self._bytes_cache: dict[int, bytes] = {}
        self.vocab_size = self.tk.get_vocab_size(with_added_tokens=True)
        ids = dict(special_ids or {})
        for key, tok in (("bos", "<bos>"), ("eos", "<eos>"), ("pad", "<pad>")):
            if key not in ids:
                got = self.tk.token_to_id(tok)
                if got is not None:
                    ids[key] = got
        self.special_ids = ids

    # ------------------------------------------------------------------------------------------ encode
    def _scan(self, text: str, parse_special: bool, plain, marks: list | None) -> list[int]:
        out: list[int] = []
        rx = self._special_re
        pos = start = 0                                 # start: where the pending ordinary text begins
        while rx is not None:
            m = rx.search(text, pos)
            if m is None:
                break
            s, e = m.span()
            lit = m.group(0)
            if not (parse_special or lit in self._always) or any(a <= s < b for a, b in plain):
                pos = s + 1                              # text here: the next literal may begin inside this one
                continue
            if s > start:
                out.extend(self._plain.encode(text[start:s], add_special_tokens=False).ids)
            out.append(self.special_tokens[lit])
            if marks is not None:
                marks.append((e, len(out)))
            pos = start = e
        if start < len(text):
            out.extend(self._plain.encode(text[start:], add_special_tokens=False).ids)
        return out

    def encode(self, text: str, parse_special: bool = False, plain=()) -> list[int]:
        return self._scan(text, parse_special, plain, None)

    def encode_marked(self, text: str, parse_special: bool = False, plain=()):
        """encode() and its resume points: (end offset in `text`, ids so far) after every special-token match."""
        marks: list[tuple[int, int]] = []
        return self._scan(text, parse_special, plain, marks), marks

    # ------------------------------------------------------------------------------------------ decode
    def token_bytes(self, i: int) -> bytes:
        """The raw bytes of one token: `<0xHH>` is that byte, "▁" is a space, a special token is its text."""
        b = self._bytes_cache.get(i)
        if b is None:
            piece = self.tk.id_to_token(int(i))
            if piece is None:
                raise IndexError("token id %d is outside the vocabulary (%d)" % (i, self.vocab_size))
            m = _BYTE_TOKEN.match(piece)
            if piece in self.special_tokens:
                b = piece.encode("utf-8")
            elif m:
                b = bytes([int(m.group(1), 16)])
            else:
                b = piece.replace("▁", " ").encode("utf-8")
            self._bytes_cache[i] = b
        return b

    def decode(self, ids, errors: str = "replace") -> str:
        return b"".join(self.token_bytes(i) for i in ids).decode("utf-8", errors=errors)


def load(folder: str | Path) -> HFTokenizer:
    """The tokenizer in `folder` (tokenizer.json, and generation_config.json for the bos / eos ids when it is there)."""
    folder = Path(folder)
    special: dict[str, int] = {}
    gen = folder / "generation_config.json"
    if gen.exists():
        try:
            cfg = json.loads(gen.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            cfg = {}
        for key, name in (("bos", "bos_token_id"), ("eos", "eos_token_id"), ("pad", "pad_token_id")):
            v = cfg.get(name)
            if isinstance(v, list):                      # several end ids: the first is "eos"
                v = v[0] if v else None
            if isinstance(v, int) and not isinstance(v, bool):
                special[key] = v
    return HFTokenizer(folder / "tokenizer.json", special)
