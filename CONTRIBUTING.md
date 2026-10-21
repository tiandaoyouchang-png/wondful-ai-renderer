# Contributing to Wondful AI Renderer

Thanks for your interest in improving Wondful AI Renderer! Issues and pull requests are welcome: new scene presets, more providers, better acceptance algorithms, case studies for other product categories, docs and translations. First-time contributors and **Hacktoberfest** participants are very welcome too.

## Repository layout

```text
wondful_ai_renderer/       The Blender add-on itself (this is what gets installed)
tests/                     Offline unit tests + a mock Codex backend
tools/                     Build, lint and Blender acceptance scripts
skills/                    Car-render style-transfer skill pack (for Codex and other agents)
geometry-render-pipeline/  Geometry-constrained commercial render pipeline (V0.1 prototype)
docs/                      Case studies (CASES.md / CASES.en.md) and showcase images
```

## Requirements

- **Blender 4.3 or newer** (`bl_info` declares `"blender": (4, 3, 0)`).
- **Python 3.11** for running the offline tests and tools outside Blender (this is what CI uses).
- For **real image generation** inside Blender, one of the official CLIs installed and logged in on your machine:
  - [Codex CLI](https://github.com/openai/codex) (with an account that has image generation), or
  - Antigravity CLI (Google account login).
- Optional: `TYPESAFE_API_KEY` for the Jev semantic engine; without it the add-on falls back to local rules.

You do **not** need Codex or Antigravity to run the test suite. The offline tests use `tests/mock_codex_backend.py`, a local mock of the Codex Responses backend and OAuth token endpoint. It can also be run on its own while debugging:

```bash
python3 tests/mock_codex_backend.py --port 8765 --log /tmp/mock.jsonl --mode ok
```

## Installing from source into Blender

The add-on is the `wondful_ai_renderer/` folder. After it is enabled, open the 3D Viewport sidebar (press `N`) and look for the **Wondful AI** tab (location: `3D Viewport > Sidebar > Wondful AI`).

### Option A: build and install a ZIP

```bash
python tools/build_release.py
```

This writes `dist/Wondful-AI-Renderer-Blender-<version>.zip`, where `<version>` is read from `bl_info["version"]` in `wondful_ai_renderer/__init__.py`. The ZIP contains the `wondful_ai_renderer/` folder (skipping `__pycache__`, `.pyc` and `.pyo` files), and the script prints the path of the file it built.

Then in Blender: **Edit → Preferences → Add-ons → Install from Disk…**, pick the ZIP and enable the add-on. (Zipping the `wondful_ai_renderer/` folder by hand works the same way.)

Rebuild and reinstall the ZIP whenever you change the code.

### Option B: symlink for live development

Link the folder into Blender's user add-ons directory so your edits are picked up without reinstalling (restart Blender, or disable and re-enable the add-on, to reload):

```bash
# Linux
ln -s "$PWD/wondful_ai_renderer" ~/.config/blender/4.3/scripts/addons/wondful_ai_renderer

# macOS
ln -s "$PWD/wondful_ai_renderer" ~/Library/Application\ Support/Blender/4.3/scripts/addons/wondful_ai_renderer
```

```powershell
# Windows (run as administrator, or with Developer Mode enabled)
New-Item -ItemType SymbolicLink -Path "$env:APPDATA\Blender Foundation\Blender\4.3\scripts\addons\wondful_ai_renderer" -Target "$PWD\wondful_ai_renderer"
```

Replace `4.3` with your Blender version and create the `addons` folder if it does not exist yet. Then enable the add-on under **Edit → Preferences → Add-ons**.

## Running the tests

CI (`.github/workflows/validate.yml`) runs on every pull request and on pushes to `main` and `wondful-*` branches, using Python 3.11 on Ubuntu. To reproduce it locally, run exactly the same steps from the repository root:

```bash
# 1. Install test dependencies
python -m pip install --disable-pip-version-check --no-cache-dir numpy pyflakes

# 2. Compile Python sources (catches syntax errors)
python -m compileall -q wondful_ai_renderer tests

# 3. Check for undefined names
python tools/check_undefined_names.py

# 4. Run the offline tests
python -m unittest discover -s tests -p "test_*.py" -v

# 5. Build the installable ZIP
python tools/build_release.py
```

CI then uploads `dist/*.zip` as the `wondful-addon-zip` artifact. Please make sure all five steps pass before opening a PR.

### Optional: checks inside a real Blender

These are not run in CI, but are useful when you touch rendering or UI code:

```bash
# Headless acceptance test (without a GPU, EEVEE cannot render headless; use Cycles for the structure passes)
WONDFUL_STRUCTURE_ENGINE=CYCLES blender -b --factory-startup \
  --python tools/blender_acceptance.py -- --out /tmp/wondful_accept

# Headless panel draw check against Blender's real RNA (pass the ZIP built by tools/build_release.py)
blender -b --factory-startup --python tools/ui_draw_check.py -- dist/Wondful-AI-Renderer-Blender-<version>.zip
```

### UI translations

PR #8 introduced `bpy.app.translations` for step ① only. The later closure of
issue #4 did not establish full UI coverage. Keep the Chinese source strings and
the existing registration/unregistration in `__init__.py`; add English entries
to `i18n.py`. Use Blender's **English** language with **Translate Interface**,
**Tooltips**, and **Reports** enabled to inspect all three surfaces.

- RNA operator titles use the `Operator` context. Explicit button text, panel
  titles, property names/descriptions and enum labels use their RNA translation
  context (normally `*`). Keep both entries when a title is also explicit button text.
- Translate a formatted UI template **before** `.format(...)`. Translate a
  disclosure/step title before adding its number or suffix. Translate status text
  before wrapping or truncating it; a wrapped fragment is no longer a catalog key.
- Workers and saved settings retain canonical Chinese status text, because stage
  detection and retry logic inspect those strings. `translate_message()` is a
  display-only compatibility layer for exact catalog messages, known formatted
  templates and explicit summary separators/affixes. Numeric field constraints
  prevent a count from consuming later clauses. Only `_MESSAGE_FIELDS` may
  recursively localize nested diagnostics; names, paths, IDs and user/model text
  must stay opaque. Never pass editable prompts or reference notes to this helper.
- Add regression cases for composed messages and numeric/unit formatting. Extend
  `tools/check_i18n.py`'s declared display functions when adding a status builder.
  The audit covers UI/RNA metadata, declared status builders and diagnostic sinks;
  it is a source contract, not a claim that all strings in the repository are UI.
- Unknown external CLI/model diagnostics remain verbatim. Other interface
  languages and disabled translation switches retain source text. Generated
  descriptions and user-entered data are not automatically translated.

```bash
python tools/check_i18n.py
python -m unittest discover -s tests -p 'test_i18n*.py' -v
blender -b --factory-startup --python-exit-code 1 --python tools/blender_i18n_check.py
```

The Blender check registers the actual add-on/RNA, verifies catalog lookups,
operator contexts and language switches, then records displayed text across 70
English panel scenarios and the comparison panel. This is a **headless** check,
not a pixel screenshot or proof that every sidebar width is free of truncation.
For visual QA, use `tools/ui_screenshots.py` with a GUI display.

## Tools at a glance

| Script | What it does |
|---|---|
| `tools/build_release.py` | Builds `dist/Wondful-AI-Renderer-Blender-<version>.zip` from `wondful_ai_renderer/`. |
| `tools/check_undefined_names.py` | Runs pyflakes over `wondful_ai_renderer`, `tests` and `tools` and fails on real undefined names, ignoring false positives from Blender property declarations. |
| `tools/blender_acceptance.py` | Headless Blender acceptance test; writes `report.json` to `--out` and exits non-zero if any check fails. |
| `tools/ui_draw_check.py` | Installs the release ZIP and draws the main panel in several states, checking every layout call (keywords, icons, properties, operators) against Blender's RNA. |
| `tools/ui_screenshots.py` | Renders real screenshots of the main panel in several states (GUI mode, e.g. under Xvfb; uses ImageMagick `import`). See the script's docstring for usage. |

## Pull request guidelines

- **Keep PRs small and focused.** One fix or feature per PR is much easier to review than a large mixed change.
- **Link the issue** your PR addresses (e.g. `Closes #123`). For larger changes, consider opening an issue first to discuss the approach.
- **Describe what you tested**: which CI steps you ran locally, your Blender version and OS, and whether you tried real generation (Codex / Antigravity) or only the mock backend.
- **Add screenshots** (before/after if possible) for any change to the UI or to rendered output.
- Add or update tests in `tests/` when you change behaviour.
- Keep the existing code style and don't commit build output (`dist/`), caches, model files or other large binaries.
- For new case studies, follow the format in `docs/CASES.md`: report the measured metrics honestly, including scenes that miss the thresholds, and credit model licenses.

Hacktoberfest contributions are welcome. Thanks for helping out!
