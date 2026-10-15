import json, sys, tempfile, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_316_regressions import load_package_module  # noqa: E402
direct = load_package_module("codex_direct")

LIVE = [
    {"slug": "gpt-6-astra", "priority": 2, "visibility": "list"},
    {"slug": "gpt-6.1-sol", "priority": 0, "visibility": "list"},
    {"slug": "gpt-5.5", "priority": 8, "visibility": "hide"},
    {"slug": "codex-auto-review", "priority": 43, "visibility": "list"},
    {"slug": "gpt-5.6-terra", "priority": 6, "visibility": "list"},
]


class ModelSelectTests(unittest.TestCase):
    def setUp(self):
        direct._MODEL_CACHE.update(at=0.0, models=[])

    def test_rank_newest_first_hidden_dropped(self):
        self.assertEqual(direct._rank_models(LIVE), ["gpt-6.1-sol", "gpt-6-astra", "gpt-5.6-terra"])

    def test_chain_user_pick_first_then_latest(self):
        chain = direct.model_chain("gpt-5.6-terra", ["gpt-6.1-sol", "gpt-5.6-terra"])
        self.assertEqual(chain[:2], ["gpt-5.6-terra", "gpt-6.1-sol"])
        self.assertEqual(len(chain), len(set(chain)))

    def test_models_cache_json_fallback(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "auth.json").write_text("{}")
            (Path(d) / "models_cache.json").write_text(json.dumps({"models": LIVE}))
            self.assertEqual(direct.available_models(None, Path(d) / "auth.json")[0], "gpt-6.1-sol")

    def test_static_fallback(self):
        self.assertEqual(direct.available_models(None, None), list(direct.FALLBACK_MODELS))


if __name__ == "__main__":
    unittest.main()
