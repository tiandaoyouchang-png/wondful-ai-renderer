"""Build the installable add-on ZIP and the update manifest.

    python3 tools/build_release.py [--notes "一句中文更新说明"] [--no-manifest]

Writes dist/Wondful-AI-Renderer-Blender-<version>.zip, copies it to releases/ (the in-app updater downloads
releases/<zip> from GitHub or its mirrors) and regenerates releases/latest.json
{version, zip, sha256, size, notes}. Notes default to the CHANGELOG heading of this version.
The ZIP is deterministic (fixed timestamps, sorted entries), so rebuilding unchanged sources keeps the sha256.
"""
import argparse
import ast
import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "wondful_ai_renderer"
RELEASES = ROOT / "releases"
MANIFEST = RELEASES / "latest.json"
FIXED_TIME = (2026, 1, 1, 0, 0, 0)


def version() -> str:
    tree = ast.parse((PKG / "__init__.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "bl_info":
            info = ast.literal_eval(node.value)
            return ".".join(str(x) for x in info["version"])
    raise SystemExit("bl_info not found")


def changelog_notes(ver: str) -> str:
    try:
        text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    except OSError:
        return ""
    match = re.search(r"^##\s*" + re.escape(ver) + r"\s*·\s*(.+?)\s*(?:（\d{4}-\d{2}-\d{2}）)?\s*$", text, re.M)
    return match.group(1).strip() if match else ""


def build_zip(out: Path) -> None:
    out.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(PKG.rglob("*")):
            if path.is_dir() or "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
                continue
            info = zipfile.ZipInfo(str(Path("wondful_ai_renderer") / path.relative_to(PKG)).replace("\\", "/"),
                                   FIXED_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, path.read_bytes())


def write_manifest(zip_path: Path, ver: str, notes: str) -> dict:
    data = zip_path.read_bytes()
    manifest = {"version": ver, "zip": zip_path.name, "sha256": hashlib.sha256(data).hexdigest(),
                "size": len(data), "notes": " ".join(notes.split())[:200]}
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--notes", default=None, help="short Chinese update note shown in the banner")
    ap.add_argument("--no-manifest", action="store_true", help="only build dist/<zip>")
    args = ap.parse_args()
    ver = version()
    out = ROOT / "dist" / f"Wondful-AI-Renderer-Blender-{ver}.zip"
    build_zip(out)
    print(out)
    if args.no_manifest:
        return
    RELEASES.mkdir(exist_ok=True)
    rel = RELEASES / out.name
    shutil.copyfile(out, rel)
    notes = args.notes
    if notes is None:
        try:
            old = json.loads(MANIFEST.read_text(encoding="utf-8"))
            notes = old.get("notes", "") if old.get("version") == ver else ""
        except (OSError, ValueError):
            notes = ""
        notes = notes or changelog_notes(ver)
    manifest = write_manifest(rel, ver, notes)
    print(rel)
    print(MANIFEST, json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
