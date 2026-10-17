"""1.2.3 从参考图生成场景粗模 — stage progress (sb_progress state machine + a headless operator run).

Pure part: stage lists, deterministic bar fractions, 「分析参考图 · 已用 1:42」 text, hints, failure note, row
icons. Blender part (skipped without ``blender``): tools/sb_progress_bg.py runs the real operator with a fake
analysis (fixture JSON) and checks the stages advance in order, finish, fail on the right stage and cancel."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_316_regressions import load_package_module  # noqa: E402

SP = load_package_module("sb_progress")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BLENDER = shutil.which("blender")


class StageListTest(unittest.TestCase):
    def test_plain_and_correct(self):
        self.assertEqual(SP.stages(False, True), ["READ", "ANALYZE", "BUILD", "CAMERA", "DONE"])
        self.assertEqual(SP.stages(False, False), ["READ", "ANALYZE", "BUILD", "DONE"])
        self.assertEqual(SP.stages(True, True),
                         ["READ", "ANALYZE", "BUILD", "CAMERA", "PREVIEW", "SCORE", "CORRECT", "REBUILD", "DONE"])
        self.assertEqual([SP.label(s) for s in SP.stages(True, True)],
                         ["读取参考图", "分析参考图", "搭建粗模", "设置相机", "预览渲染", "评分", "校正分析", "重建", "完成"])

    def test_encode_decode(self):
        ids = SP.stages(True, False)
        self.assertEqual(SP.decode(SP.encode(ids)), ids)
        self.assertEqual(SP.decode(""), SP.stages(False))
        self.assertEqual(SP.decode("BOGUS"), SP.stages(False))


class FractionTest(unittest.TestCase):
    def test_monotonic_over_stages(self):
        for correct in (False, True):
            ids = SP.stages(correct, True)
            vals = [SP.fraction(ids, s, 0) for s in ids]
            self.assertEqual(vals[0], 0.0)
            self.assertEqual(vals[-1], 1.0)
            self.assertEqual(vals, sorted(vals))
            self.assertEqual(len(set(vals)), len(vals))

    def test_ai_stage_creeps_but_stays_below_next(self):
        ids = SP.stages(True, True)
        for stage in ("ANALYZE", "CORRECT"):
            nxt = SP.fraction(ids, ids[ids.index(stage) + 1], 0)
            a, b, c = (SP.fraction(ids, stage, t) for t in (0, 60, 10 ** 6))
            self.assertLess(a, b)
            self.assertLess(c, nxt)
            self.assertAlmostEqual(c, SP.fraction(ids, stage, 10 ** 7))   # capped, no runaway

    def test_non_ai_stage_is_stage_based(self):
        ids = SP.stages(False, True)
        self.assertEqual(SP.fraction(ids, "BUILD", 0), SP.fraction(ids, "BUILD", 999))
        # cached analysis: no creep either
        self.assertEqual(SP.fraction(ids, "ANALYZE", 0, cached=True), SP.fraction(ids, "ANALYZE", 999, cached=True))

    def test_unknown_stage(self):
        self.assertEqual(SP.fraction(SP.stages(False), "PREVIEW", 5), 0.0)


class TextTest(unittest.TestCase):
    def test_elapsed(self):
        self.assertEqual(SP.fmt_elapsed(0), "0:00")
        self.assertEqual(SP.fmt_elapsed(102), "1:42")
        self.assertEqual(SP.fmt_elapsed(3725), "62:05")
        self.assertEqual(SP.fmt_elapsed(-3), "0:00")
        self.assertEqual(SP.status_text("ANALYZE", 102), "分析参考图 · 已用 1:42")

    def test_hints(self):
        self.assertIn("约 3–5 分钟", SP.hint("ANALYZE"))
        self.assertTrue(SP.hint("CORRECT"))
        self.assertEqual(SP.hint("BUILD"), "")
        self.assertEqual(SP.hint("ANALYZE", cached=True), "使用已有分析结果")

    def test_failure_note_names_stage(self):
        n = SP.failure_note("CORRECT", 260, "网络超时")
        self.assertEqual(n, "场景粗模失败（校正分析 · 已用 4:20）：网络超时")
        self.assertIn("失败", n)   # the panel turns notes with 失败 red

    def test_row_icons(self):
        ids = SP.stages(False, True)
        rows = SP.row_icons(ids, "BUILD")
        self.assertEqual([r[1] for r in rows], ["CHECKMARK", "CHECKMARK", "PLAY", "RADIOBUT_OFF", "RADIOBUT_OFF"])
        self.assertEqual([r[2] for r in rows], [True, True, True, False, False])
        rows = SP.row_icons(ids, "", failed_stage="ANALYZE")
        self.assertEqual([r[1] for r in rows], ["CHECKMARK", "ERROR", "RADIOBUT_OFF", "RADIOBUT_OFF", "RADIOBUT_OFF"])
        self.assertTrue(all(r[1] == "CHECKMARK" for r in SP.row_icons(ids, "DONE")))


@unittest.skipUnless(BLENDER and os.environ.get("WONDFUL_SKIP_BLENDER_TESTS") != "1", "blender not on PATH")
class OperatorStagesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        out = Path(tempfile.mkdtemp()) / "sbp.json"
        subprocess.run([BLENDER, "-b", "--factory-startup", "--python", str(ROOT / "tools" / "sb_progress_bg.py"), "--",
                        "repo", str(HERE / "fixtures" / "scene_blockout" / "desert_v3.json"), str(out)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300)
        cls.rep = json.loads(out.read_text(encoding="utf-8"))

    def _stages(self, run):
        seq = []
        for status, stage, *_ in self.rep[run]["trace"]:
            if status == "BLOCKOUT" and (not seq or seq[-1] != stage):
                seq.append(stage)
        return seq

    def _bar(self, run):
        return [p for status, _s, p, *_ in self.rep[run]["trace"] if status == "BLOCKOUT"]

    def test_ran(self):
        self.assertTrue(self.rep.get("ok"), self.rep.get("error"))

    def test_plain(self):
        r = self.rep["plain"]
        self.assertEqual(self._stages("plain"), ["ANALYZE", "BUILD", "CAMERA"])
        self.assertEqual((r["status"], r["final_stage"], r["failed_stage"]), ("IDLE", "DONE", ""))
        self.assertEqual(self._bar("plain"), sorted(self._bar("plain")))
        self.assertEqual(r["camera"], "Wondful 参考相机")
        self.assertIn("个体块", r["note"])
        texts = [t[3] for t in r["trace"] if t[0] == "BLOCKOUT"]
        self.assertTrue(all(t.startswith("场景粗模：") and " · 已用 " in t for t in texts), texts)

    def test_correct(self):
        self.assertEqual(self._stages("correct"),
                         ["ANALYZE", "BUILD", "CAMERA", "PREVIEW", "SCORE", "CORRECT", "REBUILD"])
        self.assertEqual(self.rep["correct"]["final_stage"], "DONE")
        self.assertEqual(self._bar("correct"), sorted(self._bar("correct")))
        self.assertIn("吻合度", self.rep["correct"]["note"])

    def test_cached(self):
        self.assertEqual(self._stages("cached"), ["ANALYZE", "BUILD", "CAMERA"])
        self.assertIn("用已保存的分析", self.rep["cached"]["note"])

    def test_fail_names_stage(self):
        r = self.rep["fail"]
        self.assertEqual((r["status"], r["failed_stage"]), ("ERROR", "ANALYZE"))
        self.assertTrue(r["note"].startswith("场景粗模失败（分析参考图 · 已用 "), r["note"])

    def test_cancel(self):
        r = self.rep["cancel"]
        self.assertEqual((r["status"], r["failed_stage"], r["final_stage"]), ("IDLE", "", ""))
        self.assertTrue(r["note"].startswith("场景粗模已取消（分析参考图"), r["note"])
        self.assertLess(r["seconds"], 15)


if __name__ == "__main__":
    unittest.main()
