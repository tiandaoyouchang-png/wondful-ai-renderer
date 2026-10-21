"""English UI translations for the connect step (source-level checks, no Blender needed)."""
import importlib.util
import runpy
import re
import unittest
from pathlib import Path

PKG = Path(__file__).resolve().parents[1] / "wondful_ai_renderer"
CJK = re.compile(r"[一-鿿]")


def _load_i18n():
    # Load the file directly: importing the package pulls in bpy, which is not available here.
    spec = importlib.util.spec_from_file_location("wondful_i18n", PKG / "i18n.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ConnectStepTranslationTests(unittest.TestCase):
    def setUp(self):
        self.i18n = _load_i18n()
        self.entries = self.i18n.TRANSLATIONS["en_US"]
        self.sources = {name: (PKG / name).read_text() for name in ("ui.py", "operators.py")}

    def test_only_english_is_translated(self):
        # Chinese stays the default: no zh_HANS (or other) table means the source text is shown.
        self.assertEqual(set(self.i18n.TRANSLATIONS), {"en_US"})

    def test_every_entry_has_english_text(self):
        for (context, msgid), text in self.entries.items():
            self.assertIn(context, {self.i18n.DEFAULT, self.i18n.OPERATOR}, msgid)
            self.assertTrue(text.strip(), msgid)
            self.assertFalse(CJK.search(text), f"translation still contains Chinese: {text}")

    def test_every_msgid_appears_in_the_add_on_source(self):
        # Keys must match the literal strings in the code exactly, or Blender never looks them up.
        checks = runpy.run_path(str(PKG.parent / "tools" / "check_i18n.py"))
        messages = checks["source_messages"]()
        # This legacy status is composed from the view template + the main-view tag.
        messages.add("视角 {v0}/{v1} · {v2}" + "（主视角）")
        for (_context, msgid) in self.entries:
            self.assertIn(msgid, messages, f"no source string for translation key: {msgid}")

    def test_composed_titles_and_formatted_text_are_translated_explicitly(self):
        ui = self.sources["ui.py"]
        self.assertIn("_t = bpy.app.translations.pgettext_iface", ui)
        self.assertIn('{STEP_NUMBERS[number - 1]}  {_t(title)}', ui)
        self.assertIn('_t("找不到 {provider}").format(provider=provider_label)', ui)
        self.assertIn((self.i18n.DEFAULT, "连接"), self.entries)
        self.assertIn((self.i18n.DEFAULT, "找不到 {provider}"), self.entries)

    def test_register_and_unregister_the_table(self):
        init = (PKG / "__init__.py").read_text()
        self.assertIn("from .i18n import TRANSLATIONS", init)
        self.assertIn("bpy.app.translations.register(__name__, TRANSLATIONS)", init)
        self.assertIn("bpy.app.translations.unregister(__name__)", init)


if __name__ == "__main__":
    unittest.main()
