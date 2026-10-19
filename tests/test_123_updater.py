"""1.2.3 更新检查 + 一键更新 — updater core (pure) + a real-Blender e2e (skipped without ``blender``).

Pure: semantic version compare, manifest parsing/validation, mirror fallback (incl. a mirror serving a wrong
file), sha/size mismatch rejection, 24 h throttle + ignore, safe extract + swap (failure leaves the install
untouched), duplicate detection. Blender: tools/e2e_update_bg.py serves a fake 9.9.9 manifest + zip over a local
HTTP server, checks the banner, runs the operators and checks the files on disk."""
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_316_regressions import load_package_module  # noqa: E402

U = load_package_module("updater")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BLENDER = shutil.which("blender")


def make_addon_zip(path, version="9.9.9", name="Wondful AI 渲染器 9.9", top="wondful_ai_renderer", extra=None):
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(f"{top}/__init__.py", f"bl_info = {{'name': {name!r}, 'version': {tuple(int(x) for x in version.split('.'))!r}}}\n")
        zf.writestr(f"{top}/marker.txt", version)
        for k, v in (extra or {}).items():
            zf.writestr(k, v)
    data = Path(path).read_bytes()
    return {"version": version, "zip": Path(path).name, "sha256": hashlib.sha256(data).hexdigest(),
            "size": len(data), "notes": "测试"}


class _Resp(io.BytesIO):
    pass


class FakeNet:
    """opener(url, timeout): per-host behaviour."""
    def __init__(self, table):
        self.table, self.calls = table, []

    def __call__(self, url, timeout):
        self.calls.append(url)
        for prefix, value in self.table.items():
            if url.startswith(prefix):
                if isinstance(value, Exception):
                    raise value
                return _Resp(value)
        raise OSError("no route")


class VersionTest(unittest.TestCase):
    def test_semantic(self):
        self.assertTrue(U.is_newer("1.2.10", "1.2.9"))
        self.assertFalse(U.is_newer("1.2.9", "1.2.10"))
        self.assertTrue(U.is_newer("1.3", "1.2.99"))
        self.assertTrue(U.is_newer("2.0.0", (1, 9, 9)))
        self.assertFalse(U.is_newer("1.2.3", (1, 2, 3)))
        self.assertFalse(U.is_newer("1.2", "1.2.0"))
        self.assertTrue(U.is_newer("v1.2.4", "1.2.3"))
        self.assertFalse(U.is_newer("garbage", "1.2.3"))
        self.assertFalse(U.is_newer("", "1.2.3"))
        self.assertEqual(U.version_str((1, 2, 3)), "1.2.3")
        self.assertEqual(U.parse_version("1.2"), (1, 2, 0))
        with self.assertRaises(ValueError):
            U.parse_version("1.2.x")


class ManifestTest(unittest.TestCase):
    GOOD = {"version": "1.2.4", "zip": "Wondful-AI-Renderer-Blender-1.2.4.zip", "sha256": "A" * 64,
            "size": 1234, "notes": "  修复\n更新  "}

    def test_good(self):
        m = U.parse_manifest(json.dumps(self.GOOD).encode())
        self.assertEqual(m["version"], "1.2.4")
        self.assertEqual(m["sha256"], "a" * 64)
        self.assertEqual(m["notes"], "修复 更新")

    def test_bad(self):
        for patch in ({"version": "abc"}, {"zip": "../evil.zip"}, {"zip": "a/b.zip"}, {"zip": "x.exe"},
                      {"sha256": "123"}, {"size": 0}, {"size": "x"}, {"size": U.MAX_ZIP_BYTES + 1}):
            with self.subTest(patch=patch), self.assertRaises(U.ManifestError):
                U.parse_manifest(dict(self.GOOD, **patch))
        for raw in (b"not json", b"[1,2]", b"\xff\xfe"):
            with self.assertRaises(U.ManifestError):
                U.parse_manifest(raw)

    def test_repo_manifest_matches_release_zip(self):
        path = ROOT / "releases" / "latest.json"
        m = U.parse_manifest(path.read_bytes())
        zip_path = ROOT / "releases" / m["zip"]
        self.assertTrue(zip_path.is_file(), zip_path)
        U.verify_bytes(zip_path.read_bytes(), m)
        init = ROOT / "wondful_ai_renderer" / "__init__.py"
        self.assertEqual(U.version_str(U.read_bl_info(init)["version"]), m["version"])


class MirrorTest(unittest.TestCase):
    def test_default_chain(self):
        os.environ.pop("WONDFUL_UPDATE_BASE_URLS", None)
        urls = U.mirror_urls("releases/latest.json")
        self.assertEqual(urls[0], "https://raw.githubusercontent.com/tiandaoyouchang-png/wondful-ai-renderer/main/releases/latest.json")
        self.assertTrue(any("cdn.jsdelivr.net/gh/tiandaoyouchang-png/wondful-ai-renderer@main/releases/latest.json" in u for u in urls))
        self.assertEqual(len(urls), 3)

    def test_env_override(self):
        os.environ["WONDFUL_UPDATE_BASE_URLS"] = "http://127.0.0.1:1/a, http://h/{path}?x=1"
        try:
            self.assertEqual(U.mirror_urls("releases/z.zip"), ["http://127.0.0.1:1/a/releases/z.zip",
                                                               "http://h/releases/z.zip?x=1"])
        finally:
            os.environ.pop("WONDFUL_UPDATE_BASE_URLS")

    def test_fallback_to_second_then_third(self):
        good = json.dumps(ManifestTest.GOOD).encode()
        net = FakeNet({"http://a/": OSError("timed out"), "http://b/": b"<html>blocked</html>", "http://c/": good})
        m = U.fetch_manifest(opener=net, bases=["http://a", "http://b", "http://c"])
        self.assertEqual(m["version"], "1.2.4")
        self.assertEqual(m["source"], "http://c/releases/latest.json")
        self.assertEqual(len(net.calls), 3)

    def test_all_fail(self):
        net = FakeNet({"http://a/": OSError("timed out"), "http://b/": OSError("refused")})
        with self.assertRaises(U.FetchError) as ctx:
            U.fetch_manifest(opener=net, bases=["http://a", "http://b"])
        self.assertEqual(str(ctx.exception), "无法连接更新服务器（已尝试 2 个地址）")
        self.assertEqual(ctx.exception.details, ["a：超时", "b：refused"])

    def test_zip_sha_mismatch_falls_through_and_rejects(self):
        with tempfile.TemporaryDirectory() as tmp:
            zp = Path(tmp) / "W-9.9.9.zip"
            m = make_addon_zip(zp, "9.9.9")
            data = zp.read_bytes()
            tampered = data[:-1] + bytes([data[-1] ^ 1])
            net = FakeNet({"http://bad/": tampered, "http://good/": data})
            out = Path(tmp) / "dl"
            out.mkdir()
            path = U.download_zip(m, out, opener=net, bases=["http://bad", "http://good"])
            self.assertEqual(path.read_bytes(), data)
            only_bad = FakeNet({"http://bad/": tampered})
            with self.assertRaises(U.FetchError) as ctx:
                U.download_zip(m, out, opener=only_bad, bases=["http://bad"])
            self.assertIn("sha256 不一致", str(ctx.exception))
            with self.assertRaises(U.UpdateError):
                U.verify_bytes(data + b"x", m)


class ThrottleTest(unittest.TestCase):
    def test_once_per_day(self):
        now = 1_800_000_000.0
        self.assertTrue(U.should_check({}, now))
        state = U.record_check({}, None, now)
        self.assertFalse(U.should_check(state, now + 3600))
        self.assertFalse(U.should_check(state, now + 24 * 3600 - 1))
        self.assertTrue(U.should_check(state, now + 24 * 3600))
        self.assertTrue(U.should_check(state, now - 7200))        # clock set back
        self.assertTrue(U.should_check({"last_check": "junk"}, now))

    def test_state_roundtrip_pending_ignore(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(U.load_state(tmp), {})
            m = U.parse_manifest(ManifestTest.GOOD)
            state = U.record_check({}, m, 100.0)
            U.save_state(tmp, state)
            state = U.load_state(tmp)
            self.assertEqual(state["latest"]["version"], "1.2.4")
            self.assertEqual(U.pending_update(state, "1.2.3")["version"], "1.2.4")
            self.assertIsNone(U.pending_update(state, "1.2.4"))
            self.assertIsNone(U.pending_update(state, "1.2.10"))
            state["ignored"] = "1.2.4"
            self.assertIsNone(U.pending_update(state, "1.2.3"))
            state["latest"]["version"] = "1.2.5"       # a later release prompts again
            self.assertEqual(U.pending_update(state, "1.2.3")["version"], "1.2.5")
            (Path(tmp) / U.STATE_FILE).write_text("{broken", encoding="utf-8")
            self.assertEqual(U.load_state(tmp), {})


class InstallTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addons = self.tmp / "addons"
        self.target = self.addons / "wondful_ai_renderer"
        self.target.mkdir(parents=True)
        (self.target / "__init__.py").write_text("bl_info = {'name': 'Wondful AI 渲染器 1.2', 'version': (1, 2, 3)}\n")
        (self.target / "old_only.py").write_text("x = 1\n")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _snapshot(self):
        return sorted(str(p.relative_to(self.addons)) for p in self.addons.rglob("*"))

    def test_swap(self):
        zp = self.tmp / "new.zip"
        make_addon_zip(zp, "9.9.9")
        info = U.install_zip(zp, self.target, "9.9.9")
        self.assertTrue(info["backup_removed"])
        self.assertEqual((self.target / "marker.txt").read_text(), "9.9.9")
        self.assertFalse((self.target / "old_only.py").exists())
        self.assertEqual([p.name for p in self.addons.iterdir()], ["wondful_ai_renderer"])   # no staging/backup left

    def test_failures_leave_install_untouched(self):
        before = self._snapshot()
        cases = {
            "wrong_version": dict(version="9.9.8"),
            "not_wondful": dict(name="Other Addon"),
            "wrong_top": dict(top="something_else"),
            "traversal": dict(extra={"../evil.py": "x"}),
        }
        for label, kw in cases.items():
            with self.subTest(label):
                zp = self.tmp / f"{label}.zip"
                make_addon_zip(zp, **{"version": kw.get("version", "9.9.9"), **{k: v for k, v in kw.items() if k != "version"}})
                with self.assertRaises(U.UpdateError):
                    U.install_zip(zp, self.target, "9.9.9")
                self.assertEqual(self._snapshot(), before)
                self.assertFalse((self.tmp / "evil.py").exists())
        junk = self.tmp / "junk.zip"
        junk.write_bytes(b"not a zip")
        with self.assertRaises(U.UpdateError):
            U.install_zip(junk, self.target, "9.9.9")
        self.assertEqual(self._snapshot(), before)

    def test_cleanup_leftovers(self):
        (self.addons / (U.BACKUP_PREFIX + "x")).mkdir()
        (self.addons / (U.STAGING_PREFIX + "y")).mkdir()
        self.assertEqual(len(U.cleanup_leftovers(self.addons)), 2)
        self.assertEqual([p.name for p in self.addons.iterdir()], ["wondful_ai_renderer"])


class DuplicateTest(unittest.TestCase):
    def test_find_and_remove(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            a, b = tmp / "user" / "addons", tmp / "ext" / "user_default"
            cur = a / "wondful_ai_renderer"
            for d in (cur, b / "wondful_ai_renderer", a / "Wondful-AI-Renderer-1.2.2", a / "other_addon",
                      a / ".wondful_backup_1", a / "no_init"):
                d.mkdir(parents=True)
            (cur / "__init__.py").write_text("bl_info = {'name': 'Wondful AI 渲染器 1.2', 'version': (1, 2, 3)}")
            (b / "wondful_ai_renderer" / "__init__.py").write_text("bl_info = {'name': 'Wondful AI 渲染器 1.1', 'version': (1, 1, 0)}")
            (a / "Wondful-AI-Renderer-1.2.2" / "__init__.py").write_text("bl_info = {'name': 'Wondful AI 渲染器 1.2', 'version': (1, 2, 2)}")
            (a / "other_addon" / "__init__.py").write_text("bl_info = {'name': 'Other'}")
            (a / ".wondful_backup_1" / "__init__.py").write_text("bl_info = {'name': 'Wondful AI 渲染器'}")
            dups = U.find_duplicates(cur, [a, b, a, tmp / "missing"])
            self.assertEqual(sorted(p.name for p in dups), ["Wondful-AI-Renderer-1.2.2", "wondful_ai_renderer"])
            self.assertNotIn(cur, dups)
            removed, failed = U.remove_dirs(dups)
            self.assertEqual((len(removed), failed), (2, []))
            self.assertTrue(cur.is_dir())
            self.assertTrue((a / "other_addon").is_dir())
            self.assertEqual(U.find_duplicates(cur, [a, b]), [])


@unittest.skipUnless(BLENDER, "blender not installed")
class BlenderUpdateE2E(unittest.TestCase):
    def test_e2e(self):
        out = tempfile.mkdtemp(prefix="wondful_upd_e2e_")
        env = dict(os.environ, BLENDER_USER_SCRIPTS=os.path.join(out, "scripts"),
                   BLENDER_USER_CONFIG=os.path.join(out, "config"), BLENDER_USER_EXTENSIONS=os.path.join(out, "ext"))
        env.pop("WONDFUL_UPDATE_BASE_URLS", None)
        env["NO_PROXY"] = env["no_proxy"] = ",".join(x for x in ("127.0.0.1", "localhost", env.get("NO_PROXY", "")) if x)
        proc = subprocess.run([BLENDER, "-b", "--factory-startup", "--python", str(ROOT / "tools" / "e2e_update_bg.py"),
                               "--", out], capture_output=True, text=True, timeout=600, env=env)
        tail = (proc.stdout + proc.stderr)[-6000:]
        self.assertEqual(proc.returncode, 0, tail)
        self.assertIn("E2E UPDATE OK", proc.stdout, tail)
        shutil.rmtree(out, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
