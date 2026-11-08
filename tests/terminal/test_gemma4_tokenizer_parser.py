"""serve/hf_tokenizer.py and serve/gemma4.py: a tokenizer.json read through `tokenizers`, and Gemma 4's thought channel and tool calls.

The tokenizer.json here is built in the test (byte fallback, Gemma-like specials), not the real Gemma file."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "tools", ROOT / "serve"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from tokenizers import AddedToken, Tokenizer, decoders, models, normalizers  # noqa: E402

from serve import gemma4  # noqa: E402
from serve.gemma4 import Gemma4OutputParser, parse_args, to_gemma4_force  # noqa: E402
from serve.hf_tokenizer import HFTokenizer, load  # noqa: E402

Q = '<|"|>'
CONTROL = ["<|turn>", "<turn|>", "<|channel>", "<channel|>", "<|think|>", "<|tool_call>", "<tool_call|>",
           "<|tool_response>", "<tool_response|>", "<|tool>", "<tool|>", Q]


def build_tokenizer(folder: Path) -> Path:
    vocab = {"<pad>": 0, "<eos>": 1, "<bos>": 2, "<unk>": 3}
    for b in range(256):
        vocab["<0x%02X>" % b] = len(vocab)
    for ch in "▁abcdefghijklmnopqrstuvwxyz":
        vocab[ch] = len(vocab)
    merges = []
    for a, b in (("▁", "t"), ("h", "e"), ("▁t", "he"), ("i", "n")):
        merges.append((a, b))
        vocab.setdefault(a + b, len(vocab))
    tk = Tokenizer(models.BPE(vocab, merges, unk_token="<unk>", byte_fallback=True))
    tk.normalizer = normalizers.Replace(" ", "▁")
    tk.decoder = decoders.Sequence([decoders.Replace("▁", " "), decoders.ByteFallback(), decoders.Fuse()])
    tk.add_special_tokens([AddedToken(t, special=True, normalized=False) for t in ["<pad>", "<eos>", "<bos>", "<unk>"]])
    tk.add_special_tokens([AddedToken(t, special=True, normalized=False) for t in CONTROL])
    tk.add_tokens([AddedToken("<h1>", special=False, normalized=False)])
    tk.save(str(folder / "tokenizer.json"))
    (folder / "generation_config.json").write_text(json.dumps({"bos_token_id": 2, "eos_token_id": [1, 106]}))
    return folder


class TokenizerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.tok = load(build_tokenizer(Path(cls.tmp.name)))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_ids(self):
        t = self.tok
        self.assertEqual(t.special_ids["bos"], 2)
        self.assertEqual(t.special_ids["eos"], 1)
        self.assertIn("<|turn>", t.control_tokens)
        self.assertNotIn("<h1>", t.control_tokens)
        self.assertEqual(t.ALWAYS, ("<h1>",))
        self.assertEqual(t.max_special_len, max(len(s) for s in CONTROL + ["<h1>", "<pad>", "<unk>", "<eos>", "<bos>"]))

    def test_generation_config_overrides(self):
        with tempfile.TemporaryDirectory() as d:
            build_tokenizer(Path(d))
            (Path(d) / "generation_config.json").write_text(json.dumps({"eos_token_id": 106, "bos_token_id": 2}))
            self.assertEqual(load(d).special_ids["eos"], 106)

    def test_specials_only_when_asked(self):
        t = self.tok
        on = t.encode("<|turn>user\nhi<turn|>", parse_special=True)
        off = t.encode("<|turn>user\nhi<turn|>")
        self.assertEqual(on[0], t.special_tokens["<|turn>"])
        self.assertEqual(on[-1], t.special_tokens["<turn|>"])
        self.assertNotIn(t.special_tokens["<|turn>"], off)
        self.assertEqual(t.decode(off), "<|turn>user\nhi<turn|>")        # still the same text
        self.assertEqual(t.decode(on), "<|turn>user\nhi<turn|>")

    def test_always_token_and_plain_span(self):
        t = self.tok
        self.assertEqual(t.encode("a<h1>")[-1], t.special_tokens["<h1>"])
        text = "x<|turn>y<|turn>"
        full = t.encode(text, parse_special=True)
        self.assertEqual(full.count(t.special_tokens["<|turn>"]), 2)
        quoted = t.encode(text, parse_special=True, plain=[(1, 8)])
        self.assertEqual(quoted.count(t.special_tokens["<|turn>"]), 1)
        self.assertEqual(t.decode(quoted), text)

    def test_marks_resume_exactly(self):
        t = self.tok
        text = "<bos><|turn>user\nthe thin one<turn|>\n<|turn>model\n"
        ids, marks = t.encode_marked(text, parse_special=True)
        self.assertEqual(ids, t.encode(text, parse_special=True))
        self.assertTrue(marks)
        for end, n in marks:
            self.assertEqual(ids[:n] + t.encode(text[end:], parse_special=True), ids)

    def test_round_trips(self):
        t = self.tok
        for text in ["", "hello world", "the thin in", "café über ß", "日本語", "emoji \U0001F600\U0001F680 end",
                     "line\n\ttab  two spaces"]:
            self.assertEqual(t.decode(t.encode(text)), text, text)
        text = "<|channel>thought\n<channel|>" + f"<|tool_call>call:f{{a:{Q}x{Q}}}<tool_call|>"
        self.assertEqual(t.decode(t.encode(text, parse_special=True)), text)

    def test_token_bytes_joined_is_decode(self):
        t = self.tok
        for text in ["the \U0001F600 café 日本", "<|turn>model\nhi<turn|>", "plain"]:
            ids = t.encode(text, parse_special=True)
            self.assertEqual(b"".join(t.token_bytes(i) for i in ids).decode("utf-8"), t.decode(ids))
            # a multi-byte character really is split across tokens (byte fallback)
        ids = t.encode("\U0001F600")
        self.assertGreater(len(ids), 1)
        self.assertEqual(t.token_bytes(t.tk.token_to_id("<0x41>")), b"A")
        self.assertEqual(t.token_bytes(t.tk.token_to_id("▁")), b" ")

    def test_server_detokenizer(self):
        try:
            from serve.server import Detokenizer
        except Exception as e:                          # the server needs more than this test has
            self.skipTest(str(e))
        d = Detokenizer(self.tok)
        out = "".join(d.push(i) for i in self.tok.encode("hi \U0001F600 é"))
        self.assertEqual(out, "hi \U0001F600 é")

    def test_bad_id(self):
        with self.assertRaises(IndexError):
            self.tok.token_bytes(10 ** 6)


# ------------------------------------------------------------------------------------------------ arguments
class ArgsTest(unittest.TestCase):
    def test_flat(self):
        self.assertEqual(parse_args(f"{{city:{Q}Zagreb{Q},days:3,ok:true,x:null,f:-1.5e2}}"),
                         {"city": "Zagreb", "days": 3, "ok": True, "x": None, "f": -150.0})

    def test_nested(self):
        got = parse_args(f"{{o:{{a:[1,{{b:{Q}c{Q}}},[]],e:{{}}}},l:[{Q}x{Q},{Q}y{Q}]}}")
        self.assertEqual(got, {"o": {"a": [1, {"b": "c"}, []], "e": {}}, "l": ["x", "y"]})

    def test_strings_with_syntax(self):
        s = 'a}, b:{ "q" ] [ , \\n <tool_call|> <|tool_call> :'
        self.assertEqual(parse_args(f"{{s:{Q}{s}{Q},t:{Q}{Q},n:1}}"), {"s": s, "t": "", "n": 1})

    def test_string_that_looks_like_a_number(self):
        self.assertEqual(parse_args(f"{{a:{Q}123{Q}}}"), {"a": "123"})

    def test_whitespace_trailing_comma_quoted_keys(self):
        self.assertEqual(parse_args(f'{{ a : 1 , {Q}b c{Q}: [1, 2,], "d": "e\\"f\\u0041" , }}'),
                         {"a": 1, "b c": [1, 2], "d": 'e"fA'})

    def test_empty_and_errors(self):
        self.assertEqual(parse_args("{}"), {})
        for bad in ["", "[1]", "{a:1", f"{{a:{Q}x}}", "{a}", "{a:1}}", "{:1}", "{a:,}"]:
            with self.assertRaises(ValueError, msg=bad):
                parse_args(bad)

    def test_call_end_outside_strings(self):
        s = f"call:f{{a:{Q}x<tool_call|>y{Q}}}"
        self.assertEqual(gemma4.call_end(s + "<tool_call|>"), len(s))
        self.assertEqual(gemma4.call_end(s), -1)


# ------------------------------------------------------------------------------------------------ parser
def run(chunks, tools=None, reason="stop", **kw):
    p = Gemma4OutputParser(tools=tools, **kw)
    evs = []
    for c in chunks:
        evs += p.feed(c)
    evs += p.finish(reason)
    reasoning = "".join(e.text for e in evs if e.kind == "reasoning")
    content = "".join(e.text for e in evs if e.kind == "content")
    calls = [(e.call.name, e.call.arguments) for e in evs if e.kind == "tool_call"]
    return reasoning, content, calls


def every_split(text, **kw):
    """The result for the whole text must be the result for any two-piece split and for one character at a time."""
    whole = run([text], **kw)
    for i in range(len(text) + 1):
        assert run([text[:i], text[i:]], **kw) == whole, (i, text[:i], text[i:])
    assert run(list(text), **kw) == whole
    return whole


TOOLS = [{"name": "get_weather", "description": "w", "parameters": {"type": "object", "properties": {
    "city": {"type": "string"}, "days": {"type": "integer"}}}},
    {"name": "write", "description": "w", "parameters": {"type": "object", "properties": {
        "path": {"type": "string"}, "content": {"type": "string"}, "opts": {"type": "object"}, "tags": {"type": "array"}}}}]


class ParserTest(unittest.TestCase):
    def test_empty_thought(self):
        self.assertEqual(every_split("<|channel>thought\n<channel|>The answer."), ("", "The answer.", []))

    def test_full_thought(self):
        r, c, calls = every_split("<|channel>thought\nI think.\nSecond line.<channel|>Final <b>answer</b> 1 < 2.")
        self.assertEqual(r.strip(), "I think.\nSecond line.")
        self.assertEqual(c.strip(), "Final <b>answer</b> 1 < 2.")
        self.assertEqual(calls, [])

    def test_no_channel_at_all(self):
        self.assertEqual(every_split("Just an answer."), ("", "Just an answer.", []))
        self.assertEqual(every_split("  \n Just an answer.")[1].strip(), "Just an answer.")

    def test_missing_closing_tag(self):
        r, c, calls = every_split("<|channel>thought\nthinking and then the reply, no closer")
        self.assertEqual(c.strip(), "thinking and then the reply, no closer")
        self.assertEqual(calls, [])

    def test_missing_closing_tag_cut_by_length(self):
        r, c, _ = run(["<|channel>thought\nlong thought"], reason="length")
        self.assertEqual(c, "")
        self.assertIn("long thought", r)

    def test_empty_reply(self):
        self.assertEqual(run([""]), ("", "", []))

    def test_tool_call(self):
        text = f"<|channel>thought\n<channel|><|tool_call>call:get_weather{{city:{Q}Zagreb{Q},days:3}}<tool_call|>"
        r, c, calls = every_split(text, tools=TOOLS)
        self.assertEqual((r, c.strip()), ("", ""))
        self.assertEqual(calls, [("get_weather", {"city": "Zagreb", "days": 3})])

    def test_string_argument_stays_a_string(self):
        calls = every_split(f"<|tool_call>call:get_weather{{city:{Q}123{Q}}}<tool_call|>", tools=TOOLS)[2]
        self.assertEqual(calls, [("get_weather", {"city": "123"})])

    def test_hard_strings(self):
        body = 'def f(a, b):\n    return {"k": [1, 2], \'s\': "x"}  # } , ] <b> </parameter>\n'
        text = (f"<|channel>thought\nuse write<channel|>Writing.<|tool_call>call:write{{path:{Q}a, b/{{c}}.py{Q},"
                f"content:{Q}{body}{Q},opts:{{deep:{{n:[1,2,{{z:null}}]}},on:true}},tags:[{Q}x{Q},{Q}y, z{Q}]}}<tool_call|>")
        r, c, calls = every_split(text, tools=TOOLS)
        self.assertEqual(c.strip(), "Writing.")
        self.assertEqual(len(calls), 1)
        name, args = calls[0]
        self.assertEqual(name, "write")
        self.assertEqual(args["path"], "a, b/{c}.py")
        self.assertEqual(args["content"].strip("\n"), body.strip("\n"))
        self.assertEqual(args["opts"], {"deep": {"n": [1, 2, {"z": None}]}, "on": True})
        self.assertEqual(args["tags"], ["x", "y, z"])

    def test_multiple_calls(self):
        text = (f"<|channel>thought\n<channel|><|tool_call>call:get_weather{{city:{Q}A{Q}}}<tool_call|>\n"
                f"<|tool_call>call:get_weather{{city:{Q}B{Q},days:2}}<tool_call|><|tool_response>")
        r, c, calls = every_split(text, tools=TOOLS)
        self.assertEqual(calls, [("get_weather", {"city": "A"}), ("get_weather", {"city": "B", "days": 2})])
        self.assertEqual(c.strip(), "")

    def test_call_without_arguments_and_unclosed_call(self):
        self.assertEqual(every_split("<|tool_call>call:get_weather{}<tool_call|>", tools=TOOLS)[2], [("get_weather", {})])
        # the closing tag is missing at the end of the reply, the arguments are complete
        self.assertEqual(every_split(f"<|tool_call>call:get_weather{{city:{Q}Z{Q}}}", tools=TOOLS)[2],
                         [("get_weather", {"city": "Z"})])

    def test_unreadable_call_is_text(self):
        r, c, calls = every_split("<|tool_call>call:get_weather{city:", tools=TOOLS)
        self.assertEqual(calls, [])
        self.assertIn("call:get_weather", c)

    def test_call_after_answer_text(self):
        text = f"<|channel>thought\nx<channel|>Let me check.\n<|tool_call>call:get_weather{{city:{Q}Q{Q}}}<tool_call|>"
        r, c, calls = every_split(text, tools=TOOLS)
        self.assertEqual(c.strip(), "Let me check.")
        self.assertEqual(calls, [("get_weather", {"city": "Q"})])

    def test_ordinary_angle_brackets(self):
        self.assertEqual(every_split("<|channel>thought\n<channel|>a < b <i> <|x")[1].strip(), "a < b <i> <|x")

    def test_held_text_is_reported(self):
        p = Gemma4OutputParser()
        p.feed("<|channel>thought\nabc")
        self.assertTrue(p.clean)
        p.feed("<chan")
        self.assertFalse(p.clean)

    def test_parser_with_thinking_flag_off(self):
        self.assertEqual(every_split("<|channel>thought\n<channel|>Hi", thinking=False), ("", "Hi", []))


class ForceTest(unittest.TestCase):
    def test_force(self):
        self.assertEqual(to_gemma4_force(None), None)
        self.assertEqual(to_gemma4_force("<tool_call>\n"), "<|channel>thought\n<channel|><|tool_call>call:")
        self.assertEqual(to_gemma4_force("<tool_call>\n<function=get_weather>\n"),
                         "<|channel>thought\n<channel|><|tool_call>call:get_weather{")

    def test_forced_text_then_arguments_is_read(self):
        forced = to_gemma4_force("<tool_call>\n<function=get_weather>\n")
        rest = f"city:{Q}Rome{Q}}}<tool_call|>"
        self.assertEqual(every_split(forced + rest, tools=TOOLS)[2], [("get_weather", {"city": "Rome"})])

    def test_pick_template(self):
        with tempfile.TemporaryDirectory() as d:
            good, bad = Path(d, "good.jinja"), Path(d, "bad.jinja")
            good.write_text("<|turn>user\n{{ x }}<turn|>")
            bad.write_text("<start_of_turn>user")
            self.assertEqual(gemma4.pick_template(good), (good, False))
            self.assertEqual(gemma4.pick_template(bad), (gemma4.FIXTURE_TEMPLATE, True))
            self.assertEqual(gemma4.pick_template(None), (gemma4.FIXTURE_TEMPLATE, False))


if __name__ == "__main__":
    unittest.main()
