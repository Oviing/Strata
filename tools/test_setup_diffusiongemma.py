"""Tests for DiffusionGemma in setup.py (SIDECAR_FAMILIES: an MLX model run by a Python engine on an Apple-silicon Mac) and its pieces in
tools/strata_mcp.py.  Mocked Mac, downloads, virtual environment and conversion - nothing is downloaded, installed or started.

    python -m unittest tools.test_setup_diffusiongemma
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402
import strata_mcp  # noqa: E402

FAM = setup.SIDECAR_FAMILIES["diffusiongemma"]
MAC16 = {"mac": True, "ram_gb": 16.0, "chip": "Apple M2", "os": "macOS 15.1", "why": None}
MAC64 = {**MAC16, "ram_gb": 64.0}
LINUX = {"mac": False, "ram_gb": 31.0, "chip": "x86_64", "os": "Linux",
         "why": "DiffusionGemma needs an Apple-silicon Mac (M1 or newer, macOS 14 or newer) to run on MLX; this computer runs Linux"}


def args(**kw):
    base = dict(family="diffusiongemma", model=None, check=False, yes=True, context=None, mode=None, models_dir=None, gguf_dir=None,
                host=None, api_key=None, browser=None, no_start=True, data_dir=None, port=None, vram_reserve_mib=None)
    base.update(kw)
    return argparse.Namespace(**base)


def run(fn, *a, **kw):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        try:
            return fn(*a, **kw), out.getvalue()
        except SystemExit as e:
            return e.code, out.getvalue()


class Entries(unittest.TestCase):
    def test_the_table(self):
        self.assertEqual((FAM["arch"], FAM["tag"], FAM["mac_only"]), ("diffusion_gemma", "dgemma-", True))
        self.assertEqual(list(FAM["models"]), ["mlx-q3", "mlx-q4"])
        for other in (setup.FAMILIES, setup.OWN_PATH_FAMILIES, setup.MULTI_FILE_FAMILIES):
            self.assertNotIn("diffusiongemma", other)
        self.assertNotIn("dgemma-", [d["tag"] for d in {**setup.FAMILIES, **setup.OWN_PATH_FAMILIES, **setup.MULTI_FILE_FAMILIES}.values()])

    def test_pinned_and_hashed(self):
        for repo in (setup.DGEMMA_SRC_REPO, setup.DGEMMA_Q4_REPO):
            self.assertRegex(setup.HF_REVISIONS[repo], r"^[0-9a-f]{40}$")
        self.assertRegex(FAM["hf"], r"/resolve/a7a81407613811e8ba63af92ac0d852b809e191f/$")
        self.assertRegex(FAM["source_hf"], r"/resolve/f7f5b7f5fa82ffc52addd066915886d497f5517b/$")
        self.assertIs(setup.REPO_PINS[setup.DGEMMA_Q4_REPO], setup.DGEMMA_Q4_FILES)
        self.assertIs(setup.REPO_PINS[setup.DGEMMA_SRC_REPO], setup.DGEMMA_SRC_FILES)
        for pins in (setup.DGEMMA_Q4_FILES, setup.DGEMMA_SRC_FILES):
            for name, (size, sha) in pins.items():
                self.assertGreater(size, 0, name)
                self.assertRegex(sha, r"^[0-9a-f]{64}$", name)
        self.assertEqual(setup.file_pin(FAM["hf"] + "model-00004-of-00004.safetensors")[0], 602183698)
        self.assertEqual(len(setup.DGEMMA_SRC_FILES), 12)

    def test_packages_are_pinned_exactly(self):
        for p in setup.DGEMMA_PACKAGES:
            self.assertRegex(p, r"^[a-z0-9-]+==\d+(\.\d+)+$")
        self.assertTrue(any(p.startswith("mlx-vlm==0.7.") for p in setup.DGEMMA_PACKAGES))

    def test_sizes_are_approximate_and_plain(self):
        self.assertAlmostEqual(FAM["models"]["mlx-q4"]["download_gb"], 16.6, delta=0.2)
        self.assertAlmostEqual(FAM["source_gb"], 52, delta=1)
        self.assertIn("about", FAM["models"]["mlx-q3"]["about"])
        self.assertIn("about", FAM["models"]["mlx-q4"]["about"])

    def test_the_default_size_follows_the_macs_memory(self):
        pick = lambda gb: setup.sidecar_default_model(FAM, gb)  # noqa: E731
        self.assertEqual([pick(16.0), pick(24.0), pick(31.9), pick(32.0), pick(64.0)], ["mlx-q3", "mlx-q3", "mlx-q3", "mlx-q4", "mlx-q4"])


class Plan(unittest.TestCase):
    def test_the_plan_off_a_mac_changes_nothing_and_says_so(self):
        with tempfile.TemporaryDirectory() as d:
            data = Path(d) / "data"
            lines = setup.dgemma_plan_lines(args(data_dir=str(data), model="mlx-q3"), "mlx-q3", LINUX)
            self.assertFalse(data.exists())
            text = "\n".join(lines)
            self.assertIn("Apple-silicon Mac", lines[0])
            self.assertIn("nothing was changed", lines[0])
            for word in ("BF16", "3 bits", "mlx==", "engine.py", "32,768", "diffusion"):
                self.assertIn(word, text)

    def test_the_plan_for_the_download_and_for_ar_and_for_own_files(self):
        with tempfile.TemporaryDirectory() as d:
            text = "\n".join(setup.dgemma_plan_lines(args(data_dir=d, model="mlx-q4", mode="ar", context=65536), "mlx-q4", MAC64))
            self.assertIn("4-bit copy", text)
            self.assertIn("65,536", text)
            self.assertIn("ar (one token at a time)", text)
            self.assertNotIn("plan only", text)
            own = "\n".join(setup.dgemma_plan_lines(args(data_dir=d, model="mlx-q4", gguf_dir="/x/mlx"), "mlx-q4", MAC64))
            self.assertIn("your MLX folder /x/mlx", own)

    def test_disk_needed_for_a_conversion_is_source_plus_output_plus_10(self):
        with tempfile.TemporaryDirectory() as d:
            paths = setup.dgemma_paths(args(data_dir=d), "mlx-q3", plan=True)
            left, need = setup.dgemma_needs_gb(FAM, "mlx-q3", paths, False)
            self.assertAlmostEqual(left, FAM["source_gb"], delta=0.1)
            self.assertAlmostEqual(need, left + 11.0 + 10.0, delta=0.1)
            self.assertEqual(setup.dgemma_needs_gb(FAM, "mlx-q3", paths, True), (0.0, 0.0))

    def test_check_via_the_command_line(self):
        with mock.patch.object(sys, "argv", ["setup.py", "--family", "diffusiongemma", "--model", "mlx-q3", "--check"]):
            code, out = run(setup.main)
        self.assertEqual(code, 1)                                   # off a Mac: not an installable computer
        self.assertIn("Apple-silicon Mac", out)


class Arguments(unittest.TestCase):
    def parse(self, *argv):
        with mock.patch.object(sys, "argv", ["setup.py", *argv]), mock.patch.object(setup, "sidecar_setup", lambda a: a) as m:
            code, _out = run(setup.main)
        return code

    def test_mode_is_read(self):
        a = self.parse("--family", "diffusiongemma", "--model", "mlx-q3", "--mode", "ar")
        self.assertEqual((a.family, a.model, a.mode), ("diffusiongemma", "mlx-q3", "ar"))
        self.assertEqual(self.parse("--family", "diffusiongemma", "--model", "mlx-q4").mode, None)
        self.assertEqual(self.parse("--family", "diffusiongemma", "--model-path", "/x").gguf_dir, "/x")

    def test_bad_mode_and_wrong_family_are_refused(self):
        for argv in (["--family", "diffusiongemma", "--mode", "fast"], ["--family", "qwen", "--mode", "ar"],
                     ["--family", "diffusiongemma", "--model", "IQ3_S"]):
            with mock.patch.object(sys, "argv", ["setup.py", *argv]), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as e:
                    setup.main()
            self.assertEqual(e.exception.code, 2, argv)


class Install(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.home = self.root / "home"
        self.home.mkdir()
        self.data = self.root / "data"
        self.calls = []
        for target, repl in (("home", lambda: self.home), ("dgemma_run", self.fake_run), ("dgemma_fetch", self.fake_fetch),
                             ("dgemma_convert", self.fake_convert), ("dgemma_free_gb", lambda p: 500.0)):
            p = mock.patch.object(setup, target, repl)
            p.start()
            self.addCleanup(p.stop)

    def fake_run(self, cmd, quiet=False):
        self.calls.append([str(c) for c in cmd])
        if "venv" in cmd:
            py = setup.dgemma_venv_python(Path(str(cmd[-1])))
            py.parent.mkdir(parents=True, exist_ok=True)
            py.write_text("#!/bin/sh\n")

    def fake_fetch(self, base, folder, pins, small, what):
        self.calls.append(["fetch", base, str(folder)])
        folder.mkdir(parents=True, exist_ok=True)
        for n in (*pins, *small):
            (folder / n).write_text("x")
        (folder / "config.json").write_text("{}")
        (folder / "model-00001.safetensors").write_text("x")
        (folder / "chat_template.jinja").unlink(missing_ok=True)       # the fallback to Strata's own template is used and said

    def fake_convert(self, py, source, out, bits):
        self.calls.append(["convert", str(source), str(out), str(bits)])
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.json").write_text("{}")
        (out / "model.safetensors").write_text("x")

    def go(self, host, **kw):
        with mock.patch.object(setup, "dgemma_host", lambda: host):
            return run(setup.install_diffusiongemma, args(data_dir=str(self.data), **kw))

    def config(self, model):
        return json.loads((self.home / f"strata-dgemma-{model}.json").read_text(encoding="utf-8"))

    def test_off_a_mac_it_refuses_in_plain_words_and_writes_nothing(self):
        code, out = self.go(LINUX)
        self.assertEqual(code, 1)
        self.assertIn("Apple-silicon Mac", out)
        self.assertIn("Nothing was changed", out)
        self.assertEqual(self.calls, [])
        self.assertEqual(list(self.home.iterdir()), [])

    def test_too_little_memory_is_refused(self):
        code, out = self.go({**MAC16, "ram_gb": 8.0})
        self.assertEqual(code, 1)
        self.assertIn("too little", out)
        self.assertEqual(self.calls, [])

    def test_not_enough_disk_is_refused_before_anything_is_fetched(self):
        with mock.patch.object(setup, "dgemma_free_gb", lambda p: 30.0):
            code, out = self.go(MAC16)
        self.assertEqual(code, 1)
        self.assertIn("not enough free disk space", out)
        self.assertEqual(self.calls, [])

    def test_the_3bit_install_on_a_16gb_mac(self):
        code, out = self.go(MAC16)
        self.assertEqual(code, 0, out)
        cfg = self.config("mlx-q3")
        runtime = self.data / "runtimes" / "diffusiongemma-mlx"
        self.assertEqual(cfg["arch"], "diffusion_gemma")
        self.assertEqual(cfg["model_name"], "diffusiongemma-26b-a4b-it")
        self.assertEqual(cfg["mode"], "diffusion")
        self.assertEqual(cfg["port"], 8080)
        self.assertEqual(Path(cfg["exe"]), runtime / "strata-dgemma")
        a = cfg["args"]
        self.assertEqual(a[0], "--serve")
        self.assertEqual(a[a.index("--model") + 1], str(self.data / "models" / "dgemma-mlx-q3"))
        self.assertEqual(a[a.index("--max-context") + 1], "32768")           # the default on a 16 GB Mac
        self.assertEqual(a[a.index("--mode") + 1], "diffusion")
        self.assertEqual(a[a.index("--stop-ids") + 1], "1,106")
        self.assertEqual(cfg["tokenizer"], str(self.data / "packs" / "dgemma-mlx-q3" / "tokenizer"))
        launcher = (runtime / "strata-dgemma").read_text()
        self.assertEqual(launcher, f'#!/bin/sh\nexec "{runtime}/bin/python" -I "{ROOT}/tools/diffusiongemma/engine.py" "$@"\n')
        tok = Path(cfg["tokenizer"])
        for name in ("tokenizer.json", "generation_config.json", "chat_template.jinja"):
            self.assertTrue((tok / name).is_file(), name)
        self.assertIn("{{", (tok / "chat_template.jinja").read_text(encoding="utf-8"))          # Strata's own template, the source had none here
        pip = next(c for c in self.calls if "pip" in c)
        self.assertEqual(pip[-3:], list(setup.DGEMMA_PACKAGES))
        convert = next(c for c in self.calls if c[0] == "convert")
        self.assertEqual(convert[3], "3")
        self.assertFalse((self.data / "models" / "dgemma-source").exists(), "--yes deletes the BF16 download")
        self.assertTrue((self.home / "run-dgemma-mlx-q3.sh").is_file())

    def test_the_source_is_kept_without_a_yes(self):
        with mock.patch.object(sys.stdin, "isatty", lambda: False):
            code, out = self.go(MAC16, yes=False)
        self.assertEqual(code, 0, out)
        self.assertTrue((self.data / "models" / "dgemma-source").is_dir())
        self.assertIn("--yes deletes it", out)

    def test_the_4bit_install_ar_mode_context_and_port(self):
        code, out = self.go(MAC64, model="mlx-q4", mode="ar", context=65536, port=9001)
        self.assertEqual(code, 0, out)
        cfg = self.config("mlx-q4")
        self.assertEqual((cfg["mode"], cfg["port"]), ("ar", 9001))
        self.assertEqual(cfg["args"][cfg["args"].index("--mode") + 1], "ar")
        self.assertEqual(cfg["args"][cfg["args"].index("--max-context") + 1], "65536")
        self.assertFalse(any(c[0] == "convert" for c in self.calls))
        self.assertTrue(any(c[0] == "fetch" and c[1] == FAM["hf"] for c in self.calls))

    def test_your_own_mlx_folder_is_used_where_it_is(self):
        own = self.root / "mine"
        own.mkdir()
        (own / "config.json").write_text("{}")
        (own / "w.safetensors").write_text("x")
        (own / "tokenizer.json").write_text("{}")
        code, out = self.go(MAC64, model="mlx-q4", gguf_dir=str(own))
        self.assertEqual(code, 0, out)
        cfg = self.config("mlx-q4")
        self.assertEqual(cfg["args"][cfg["args"].index("--model") + 1], str(own))
        self.assertFalse(any(c[0] in ("fetch", "convert") for c in self.calls))

    def test_a_folder_that_is_not_an_mlx_model_is_refused(self):
        bad = self.root / "bad"
        bad.mkdir()
        code, out = self.go(MAC64, model="mlx-q4", gguf_dir=str(bad))
        self.assertEqual(code, 1)
        self.assertIn("does not look like an MLX model folder", out)

    def test_a_second_run_does_not_make_the_environment_again(self):
        self.go(MAC16)
        n = len([c for c in self.calls if "venv" in c])
        self.go(MAC16)
        self.assertEqual(len([c for c in self.calls if "venv" in c]), n)
        self.assertEqual(len([c for c in self.calls if "pip" in c]), 1)


class Fetch(unittest.TestCase):
    def test_small_files_are_checked_by_size_and_large_ones_by_hash(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            def fake_download(url, dst, what=None):
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(b"abc")
            seen = []
            with mock.patch.object(setup, "download", fake_download), mock.patch.object(setup, "verify_sha256", lambda p, size, sha, **k: seen.append((p.name, size, sha))):
                setup.dgemma_fetch("https://h/r/resolve/" + "a" * 40 + "/", folder, {"big.safetensors": (3, "0" * 64)}, {"c.json": 3}, "x")
                self.assertEqual(seen, [("big.safetensors", 3, "0" * 64)])
                code, _out = run(setup.dgemma_fetch, "https://h/r/resolve/" + "a" * 40 + "/", folder, {}, {"d.json": 4}, "x")
                self.assertEqual(code, 1)
                self.assertFalse((folder / "d.json").exists())


class McpRows(unittest.TestCase):
    def test_the_tables_carry_the_sidecar_family(self):
        s = _server()
        models, families, _ctx, _src = s.tables()
        self.assertTrue(families["diffusiongemma"]["sidecar"])
        self.assertEqual(families["diffusiongemma"]["tag"], "dgemma-")
        self.assertEqual(s.sizes_of(models, "diffusiongemma"), ["mlx-q3", "mlx-q4"])
        self.assertIn("mlx-q3", s.own_size_names())
        self.assertTrue(s.own_families()["diffusiongemma"]["mac_only"])
        self.assertEqual(s.own_size("diffusiongemma", "mlx-q4")["download_gb"], FAM["models"]["mlx-q4"]["download_gb"])

    def test_the_words_and_the_recommendation_by_memory(self):
        s = _server()
        pc = {"ram_gb": 64.0, "gpus": []}
        self.assertTrue(s.sidecar_words(pc, "diffusiongemma", "mlx-q3").startswith("does not fit: needs an Apple-silicon Mac"))
        self.assertIsNone(s.sidecar_recommend(pc)["model"])
        for ram, model, words in ((16.0, "mlx-q3", "fits"), (16.0, "mlx-q4", "tight"), (24.0, "mlx-q3", "fits"), (32.0, "mlx-q4", "fits"), (8.0, "mlx-q3", "does not fit")):
            mac = {"ram_gb": ram, "mac": {"chip": "Apple M2"}}
            self.assertTrue(s.sidecar_words(mac, "diffusiongemma", model).startswith(words), (ram, model))
        self.assertEqual(s.sidecar_recommend({"ram_gb": 16.0, "mac": {"chip": "Apple M2"}})["model"], "mlx-q3")
        self.assertEqual(s.sidecar_recommend({"ram_gb": 48.0, "mac": {"chip": "Apple M2"}})["model"], "mlx-q4")

    def test_the_model_table_lists_it_on_a_pc_and_on_a_mac(self):
        s = _server()
        pc = s.model_table({"ram_gb": 64.0, "gpus": []})
        row = next(f for f in pc if f["family"] == "diffusiongemma")
        self.assertEqual([x["model"] for x in row["sizes"]], ["mlx-q3", "mlx-q4"])
        self.assertTrue(all(x["on_this_pc"].startswith("does not fit") and x["sidecar"] for x in row["sizes"]))
        s, hw = _mac(32)
        row = next(f for f in s.model_table(hw) if f["family"] == "diffusiongemma")
        self.assertEqual({x["model"]: x["on_this_pc"] for x in row["sizes"]}, {"mlx-q3": "fits", "mlx-q4": "fits"})
        self.assertEqual(s.recommend(hw)["sidecar"]["model"], "mlx-q4")
        s, hw = _mac(16)
        row = next(f for f in s.model_table(hw) if f["family"] == "diffusiongemma")
        self.assertEqual({x["model"]: x["on_this_pc"].split(":")[0] for x in row["sizes"]}, {"mlx-q3": "fits", "mlx-q4": "tight"})
        self.assertEqual(s.recommend(hw)["sidecar"]["model"], "mlx-q3")

    def test_expected_files_are_the_pinned_ones(self):
        with tempfile.TemporaryDirectory() as d:
            q4 = strata_mcp.expected_files(setup, "diffusiongemma", "mlx-q4", Path(d), False)
            self.assertEqual({Path(f["path"]).parent.name for f in q4}, {"dgemma-mlx-q4"})
            self.assertIn(602183698, [f["bytes"] for f in q4])
            q3 = strata_mcp.expected_files(setup, "diffusiongemma", "mlx-q3", Path(d), False)
            self.assertEqual({Path(f["path"]).parent.name for f in q3}, {"dgemma-source"})
            self.assertGreater(sum(f["bytes"] for f in q3) / 1e9, 51)

    def test_the_install_plan_is_refused_off_a_pc_and_made_on_a_mac(self):
        s = _server()
        tool = strata_mcp.Tools(s)
        with tempfile.TemporaryDirectory() as tmp:
            d = str(Path(tmp) / "Strata-data")
            with mock.patch.object(type(s), "hardware", lambda self_: {"ram_gb": 64.0, "gpus": []}):
                with self.assertRaises(strata_mcp.ToolError) as e:
                    tool.install_plan("diffusiongemma", "mlx-q3", None, None, None, None, None, None, None, None, d)
            self.assertIn("Apple-silicon Mac", str(e.exception))
            s, hw = _mac(16)
            tool = strata_mcp.Tools(s)
            with mock.patch.object(type(s), "hardware", lambda self_: hw), mock.patch.object(type(s), "disk_free", lambda self_, p: 500.0):
                plan, args = tool.install_plan("diffusiongemma", "mlx-q3", None, None, None, None, None, None, None, None, d)
                self.assertEqual(plan["context"], 32768)
                self.assertGreater(plan["disk_needed_gb"], 70)
                self.assertEqual(args[:6], ["--yes", "--no-start", "--family", "diffusiongemma", "--model", "mlx-q3"])
                self.assertNotIn("--vision", args)
                self.assertIsNone(plan["disk_short"])


def _server():
    return strata_mcp.Strata(ROOT)


def _mac(ram):
    """The Strata catalog as a Mac with `ram` GB of unified memory sees itself (the same mock as tools/test_strata_mcp.py MacCatalog)."""
    from strata_control import mac_fit
    s = _server()
    info = {"chip": "Apple M4", "ram_gb": float(ram), "macos": "27.2", "arm64": True, "clt": "", "perf_cores": 4}
    with mock.patch.object(strata_mcp.sys, "platform", "darwin"), mock.patch.object(mac_fit, "mac_info", return_value=info):
        hw = s._hardware()
    return s, hw


if __name__ == "__main__":
    unittest.main()
