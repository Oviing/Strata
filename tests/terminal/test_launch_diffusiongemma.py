"""`strata launch` for DiffusionGemma (a sidecar family: MLX on an Apple-silicon Mac): the two sizes are in the list, `--plan` works on any
computer and says it needs a Mac, `--mode` is read, and a launch off a Mac is refused in plain words.  Nothing is downloaded or started."""

from __future__ import annotations

import contextlib
import json
import sys
import unittest
from unittest import mock

from strata_cli.launch import settings as S
from strata_cli.launch.catalog import Catalog
from strata_cli.launch.cmd import settings_from_args
from tests.terminal.launch_support import HW_12, Home, catalog, run_cli

MAC_HW = {"ram_gb": 16.0, "os": "Darwin", "cpu": {"avx2": None, "name": "Apple M2"}, "mac": {"chip": "Apple M2", "macos": "15.1", "unified_gb": 16.0},
          "gpus": [{"index": 0, "name": "Apple M2", "vendor": "apple", "vram_gb": 0.0, "unified_gb": 16.0, "usable": True}]}


class LaunchDiffusionGemma(unittest.TestCase):
    def setUp(self):
        self.home = Home().__enter__()
        self.addCleanup(self.home.__exit__, None, None, None)
        self.hw = dict(HW_12)
        p = mock.patch.object(Catalog, "hardware", lambda s, fresh=False: dict(self.hw))
        p.start()
        self.addCleanup(p.stop)

    def a_mac(self):
        """This computer as an M2 Mac with 16 GB, for setup.dgemma_host (the same mock as tools/test_strata_mcp.py MacCatalog)."""
        from strata_control import mac_fit
        info = {"chip": "Apple M2", "ram_gb": 16.0, "macos": "15.1", "arm64": True, "clt": "", "perf_cores": 4}
        stack = contextlib.ExitStack()
        stack.enter_context(mock.patch.object(sys, "platform", "darwin"))
        stack.enter_context(mock.patch.object(mac_fit, "mac_info", return_value=info))
        return stack

    def test_the_sizes_are_in_the_list_and_resolve(self):
        cat = catalog(self.hw)
        rows = {r.model: r for r in cat.rows() if r.family == "diffusiongemma"}
        self.assertEqual(list(rows), ["mlx-q3", "mlx-q4"])
        self.assertEqual(rows["mlx-q4"].id, "dgemma-mlx-q4")
        self.assertAlmostEqual(rows["mlx-q4"].download_gb, 16.6, delta=0.2)
        self.assertEqual(rows["mlx-q3"].verdict, "no")                      # a PC: it needs a Mac (said, not hidden)
        self.assertIn("Apple-silicon Mac", rows["mlx-q3"].words)
        self.assertFalse(rows["mlx-q3"].images)
        self.assertEqual(cat.resolve("mlx-q3"), ("diffusiongemma", "mlx-q3"))
        self.assertEqual(cat.resolve("MLX-Q4", "diffusiongemma"), ("diffusiongemma", "mlx-q4"))
        with self.assertRaises(ValueError) as e:
            cat.resolve("mlx-q3", "qwen")
        self.assertIn("has no size", str(e.exception))
        self.assertEqual(cat.max_context("diffusiongemma", "mlx-q3"), 32768)
        self.assertEqual(cat.tag_of("diffusiongemma", "mlx-q3"), "dgemma-mlx-q3")

    def test_on_a_mac_the_words_follow_the_memory(self):
        self.hw = dict(MAC_HW)
        rows = {r.model: r for r in catalog(self.hw).rows() if r.family == "diffusiongemma"}
        self.assertEqual((rows["mlx-q3"].verdict, rows["mlx-q4"].verdict), ("fits", "tight"))
        self.assertEqual(catalog(self.hw).sidecar_default("diffusiongemma"), "mlx-q3")
        self.hw = {**MAC_HW, "ram_gb": 48.0}
        self.assertEqual(catalog(self.hw).sidecar_default("diffusiongemma"), "mlx-q4")

    def test_the_other_families_are_unchanged(self):
        fams = {r.family for r in catalog(self.hw).rows()}
        self.assertIn("qwen", fams)
        self.assertIn("deepseek41", fams)
        self.assertEqual(catalog(self.hw).resolve("iq3_xxs"), ("qwen", "IQ3_XXS"))

    def test_the_plan_off_a_mac_says_so_and_changes_nothing(self):
        code, out, err = run_cli(["launch", "--plan", "--family", "diffusiongemma", "--model", "mlx-q3"])
        self.assertEqual(code, 0, err + out)
        self.assertIn("Apple-silicon Mac", out)
        self.assertIn("Nothing was changed.", out)
        self.assertIn("BF16", out)
        self.assertIn("same as   strata launch --model mlx-q3 --family diffusiongemma", out)
        self.assertEqual(list(self.home.base.glob("install/*")), [])

    def test_the_plan_as_json_with_mode_and_context(self):
        code, out, err = run_cli(["launch", "--plan", "--output", "json", "--model", "mlx-q4", "--mode", "ar", "--context", "64k"])
        self.assertEqual(code, 0, err + out)
        plan = json.loads(out)["plan"]
        self.assertEqual((plan["family"], plan["model"], plan["mode"], plan["context"]), ("diffusiongemma", "mlx-q4", "ar", 65536))
        self.assertFalse(plan["runs_here"])
        self.assertFalse(plan["changed"])
        self.assertIn("--mode ar", plan["same_as"])
        self.assertIn("ar (one token at a time)", "\n".join(plan["lines"]))

    def test_family_alone_takes_the_size_for_this_computer(self):
        code, out, err = run_cli(["launch", "--plan", "--output", "json", "--family", "diffusiongemma"])
        self.assertEqual(code, 0, err + out)
        self.assertEqual(json.loads(out)["plan"]["model"], "mlx-q4")        # the PC here has 64 GB of memory

    def test_the_plan_on_a_mac(self):
        self.hw = dict(MAC_HW)
        with self.a_mac():
            code, out, err = run_cli(["launch", "--plan", "--output", "json", "--family", "diffusiongemma"])
        self.assertEqual(code, 0, err + out)
        plan = json.loads(out)["plan"]
        self.assertEqual(plan["model"], "mlx-q3")
        self.assertTrue(plan["runs_here"])
        self.assertNotIn("plan only", "\n".join(plan["lines"]))

    def test_a_launch_off_a_mac_is_refused_in_plain_words(self):
        with mock.patch("subprocess.call") as call:
            code, out, err = run_cli(["launch", "--family", "diffusiongemma", "--model", "mlx-q3", "--no-tui", "--yes"])
        self.assertEqual(code, 5)
        self.assertIn("Apple-silicon Mac", err)
        self.assertIn("Nothing was changed", err)
        call.assert_not_called()

    def test_a_launch_on_a_mac_hands_over_to_setup_with_the_mode(self):
        self.hw = dict(MAC_HW)
        with mock.patch.object(Catalog, "sidecar_runs_here", lambda s: True), mock.patch("subprocess.call", return_value=0) as call:
            code, out, err = run_cli(["launch", "--family", "diffusiongemma", "--model", "mlx-q3", "--mode", "ar", "--no-tui", "--yes", "--port", "9100"])
        self.assertEqual(code, 0, err + out)
        argv = call.call_args.args[0]
        self.assertTrue(argv[1].endswith("setup.py"))
        self.assertEqual(argv[argv.index("--family") + 1], "diffusiongemma")
        self.assertEqual(argv[argv.index("--mode") + 1], "ar")
        self.assertEqual(argv[argv.index("--context") + 1], "32768")
        self.assertEqual(argv[argv.index("--port") + 1], "9100")
        self.assertIn("--yes", argv)

    def test_mode_is_a_setting_and_a_flag(self):
        self.assertEqual(S.GEN_MODES, ("diffusion", "ar"))
        s = S.LaunchSettings(family="diffusiongemma", model="mlx-q3", mode="ar")
        self.assertIn("--mode ar", S.flags_text(s))
        self.assertEqual(S.LaunchSettings.from_dict(s.to_dict()).mode, "ar")
        self.assertNotIn(" --mode ", S.flags_text(S.LaunchSettings(model="iq3_xxs")))
        code, _out, err = run_cli(["launch", "--plan", "--model", "mlx-q3", "--mode", "fast"])
        self.assertEqual(code, 2)
        self.assertIn("decoding mode", err)

    def test_the_settings_carry_the_mode(self):
        import argparse
        from strata_cli.launch.cmd import add_parsers
        ap = argparse.ArgumentParser()
        add_parsers(ap.add_subparsers(dest="cmd"))
        a = ap.parse_args(["launch", "--model", "mlx-q3", "--mode", "ar"])
        s, _notes = settings_from_args(a, catalog(self.hw))
        self.assertEqual((s.family, s.model, s.mode), ("diffusiongemma", "mlx-q3", "ar"))


if __name__ == "__main__":
    unittest.main()
