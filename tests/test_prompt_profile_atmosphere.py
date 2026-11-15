"""Project prompt-pack fidelity and content-based cache invalidation after reload."""
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_316_regressions import load_package_module

PP = load_package_module("prompt_profiles")
PE = load_package_module("prompt_engine")


class ProfileAtmosphereTests(unittest.TestCase):
    def tearDown(self):
        PP._profile_cached.cache_clear()

    def test_appended_common_pack_preserves_reference_exposure_and_effects(self):
        for provider in PP.PROFILES:
            with self.subTest(provider=provider):
                text = PP.system_prompt(PE.REPLICA_POLISH_SYSTEM_PROMPT, provider)
                common = text.split("--- Versioned Wondful art direction / target profile ---", 1)[1]
                self.assertIn(PE.REFERENCE_ATMOSPHERE_POLICY, text)
                for rule in ("source-faithful", "dramatic, dark or high-key exposure",
                             "intentional shadow-detail loss", "relative subject/background exposure",
                             "effect/motion type, direction and strength", "count as requested",
                             "Daytime/static references stay daytime/static", "No unrequested smoke"):
                    self.assertIn(rule, common)
                self.assertNotIn("Produce a restrained", common)
                self.assertNotIn("readable shadow details", common)
                self.assertNotIn("keeps shadow-side features readable", common)

    def test_each_common_pack_edit_refreshes_both_provider_cache_fingerprints_on_reload(self):
        source = PP.ROOT
        for changed in ("SKILL.md", "references/commercial.md"):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "pack"
                shutil.copytree(source, root)
                with mock.patch.object(PP, "ROOT", root):
                    PP._profile_cached.cache_clear()
                    before = {provider: PP.metadata(provider, "model") for provider in PP.PROFILES}
                    cache = {provider: PP.style_cache_key("same-image-bytes", ["same-note"], provider, "model")
                             for provider in PP.PROFILES}
                    changed_path = root / changed
                    changed_path.write_text(changed_path.read_text(encoding="utf-8") +
                                            "\nPreserve this source's observed exposure.\n", encoding="utf-8")
                    PP._profile_cached.cache_clear()  # fresh plugin/replay loads the edited pack
                    after = {}
                    for provider, old in before.items():
                        refreshed = PP.metadata(provider, "model")
                        after[provider] = refreshed
                        self.assertNotEqual(old["sha256"], refreshed["sha256"])
                        self.assertNotEqual(old["target_key"], refreshed["target_key"])
                        self.assertNotEqual(cache[provider], PP.style_cache_key(
                            "same-image-bytes", ["same-note"], provider, "model"))
                        props = SimpleNamespace(last_ai_prompt="polished", analysis_provider=provider,
                                                prompt_target_key=old["target_key"])
                        prefs = SimpleNamespace(codex_model="model", antigravity_model="model")
                        self.assertTrue(PP.needs_target_refresh(props, prefs))
                        props.prompt_target_key = refreshed["target_key"]
                        self.assertFalse(PP.needs_target_refresh(props, prefs))
                    PP._profile_cached.cache_clear()
                    for provider in PP.PROFILES:
                        self.assertEqual(after[provider], PP.metadata(provider, "model"))


if __name__ == "__main__":
    unittest.main()
