"""1.2.3 更新检查 + 一键更新 — pure core (no ``bpy``; the Blender side is ``update_ops.py``).

Update source: ``releases/latest.json`` in the GitHub repo
    {"version": "1.2.4", "zip": "Wondful-AI-Renderer-Blender-1.2.4.zip",
     "sha256": "<64 hex>", "size": 441604, "notes": "一句中文说明"}
fetched over a mirror chain (GitHub raw → jsDelivr → ghproxy-style mirror) with short timeouts; the zip is
fetched over the same chain from ``releases/<zip>`` and must match ``sha256`` + ``size`` before anything on disk
is touched. Installing extracts into a hidden staging dir beside the loaded add-on, validates it
(top-level ``wondful_ai_renderer/__init__.py`` whose ``bl_info`` version equals the manifest), then swaps
directories with the old copy kept as a hidden backup until the swap succeeded. Hidden (dot) dirs are never
scanned by Blender's add-on loader, so staging/backup can never show up as a second add-on.

Testing: ``WONDFUL_UPDATE_BASE_URLS`` (comma separated base URLs; ``{path}`` placeholder optional, otherwise
the path is appended) replaces the mirror chain; ``WONDFUL_UPDATE_STATE_DIR`` replaces the state dir.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shutil
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

PACKAGE_DIR_NAME = "wondful_ai_renderer"
REPO = "tiandaoyouchang-png/wondful-ai-renderer"
BRANCH = "main"
MANIFEST_PATH = "releases/latest.json"
RELEASES_PAGE = f"https://github.com/{REPO}/releases"
REPO_PAGE = f"https://github.com/{REPO}"
# Order matters: GitHub raw first (always fresh), then China-reachable mirrors.
MIRRORS = (
    "https://raw.githubusercontent.com/" + REPO + "/" + BRANCH + "/{path}",
    "https://cdn.jsdelivr.net/gh/" + REPO + "@" + BRANCH + "/{path}",
    "https://ghfast.top/https://raw.githubusercontent.com/" + REPO + "/" + BRANCH + "/{path}",
)
MANIFEST_TIMEOUT = 6.0          # seconds per mirror
DOWNLOAD_TIMEOUT = 20.0         # seconds per mirror (socket timeout, not total)
CHECK_INTERVAL = 24 * 3600      # auto check at most once per 24 h
MAX_MANIFEST_BYTES = 64 * 1024
MAX_ZIP_BYTES = 50 * 1024 * 1024
USER_AGENT = "WondfulAIRenderer-Updater"
STATE_FILE = "update_state.json"
STAGING_PREFIX = ".wondful_update_"
BACKUP_PREFIX = ".wondful_backup_"

_VERSION_RE = re.compile(r"^\s*v?(\d+(?:\.\d+){0,3})\s*$")
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_ZIP_RE = re.compile(r"^[A-Za-z0-9._-]+\.zip$")


class UpdateError(Exception):
    """Any failure the user should read (Chinese message)."""


class ManifestError(UpdateError):
    pass


class FetchError(UpdateError):
    details: list = []


# ---------------------------------------------------------------------------
# versions
# ---------------------------------------------------------------------------

def parse_version(value) -> tuple:
    """'1.2.10' / 'v1.2' / (1, 2, 3) → (1, 2, 10) padded to 3 parts; ValueError otherwise."""
    if isinstance(value, (tuple, list)):
        parts = tuple(int(x) for x in value)
    else:
        match = _VERSION_RE.match(str(value or ""))
        if not match:
            raise ValueError(f"不是版本号：{value!r}")
        parts = tuple(int(x) for x in match.group(1).split("."))
    if not parts or any(p < 0 for p in parts):
        raise ValueError(f"不是版本号：{value!r}")
    return parts + (0,) * (3 - len(parts)) if len(parts) < 3 else parts


def version_str(value) -> str:
    return ".".join(str(x) for x in parse_version(value))


def is_newer(remote, local) -> bool:
    """Semantic compare: 1.2.10 > 1.2.9; unparsable remote is never newer."""
    try:
        return parse_version(remote) > parse_version(local)
    except (ValueError, TypeError):
        return False


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------

def parse_manifest(data) -> dict:
    """Validate the manifest bytes/str/dict; returns a clean dict or raises ManifestError."""
    try:
        if isinstance(data, (bytes, bytearray)):
            data = data.decode("utf-8-sig")
        obj = json.loads(data) if isinstance(data, str) else data
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestError(f"更新信息格式错误：{exc}") from None
    if not isinstance(obj, dict):
        raise ManifestError("更新信息格式错误：不是 JSON 对象")
    try:
        version = version_str(obj.get("version"))
    except ValueError:
        raise ManifestError(f"更新信息里的版本号无效：{obj.get('version')!r}") from None
    zip_name = str(obj.get("zip") or "")
    if not _ZIP_RE.match(zip_name) or ".." in zip_name:
        raise ManifestError(f"更新信息里的文件名无效：{zip_name!r}")
    sha = str(obj.get("sha256") or "").strip().lower()
    if not _SHA_RE.match(sha):
        raise ManifestError("更新信息里的 sha256 无效")
    try:
        size = int(obj.get("size"))
    except (TypeError, ValueError):
        raise ManifestError("更新信息里的文件大小无效") from None
    if size <= 0 or size > MAX_ZIP_BYTES:
        raise ManifestError(f"更新信息里的文件大小无效：{size}")
    notes = " ".join(str(obj.get("notes") or "").split())[:200]
    return {"version": version, "zip": zip_name, "sha256": sha, "size": size, "notes": notes}


def build_manifest(zip_path, version, notes="") -> dict:
    """Used by tools/build_release.py."""
    data = Path(zip_path).read_bytes()
    return parse_manifest({"version": version, "zip": Path(zip_path).name,
                           "sha256": hashlib.sha256(data).hexdigest(), "size": len(data), "notes": notes})


# ---------------------------------------------------------------------------
# mirror chain
# ---------------------------------------------------------------------------

def mirror_urls(path: str, bases=None) -> list:
    """Every URL to try for ``path`` (repo-relative), in order. ``WONDFUL_UPDATE_BASE_URLS`` overrides."""
    if bases is None:
        env = os.environ.get("WONDFUL_UPDATE_BASE_URLS", "").strip()
        bases = [b.strip() for b in env.split(",") if b.strip()] if env else list(MIRRORS)
    urls = []
    for base in bases:
        url = base.replace("{path}", path) if "{path}" in base else base.rstrip("/") + "/" + path
        if url not in urls:
            urls.append(url)
    return urls


def _short_error(exc) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        return f"HTTP {exc.code}"
    reason = getattr(exc, "reason", None) or exc
    text = str(reason) or exc.__class__.__name__
    return ("超时" if "timed out" in text.lower() else text)[:80]


def _host(url: str) -> str:
    return url.split("//", 1)[-1].split("/", 1)[0]


def fetch_first(urls, *, timeout=MANIFEST_TIMEOUT, max_bytes=MAX_MANIFEST_BYTES, opener=None,
                progress=None, cancel=None, validate=None):
    """GET each URL in turn; return (bytes, url) from the first that answers (and passes ``validate``).
    ``opener(url, timeout)`` returns a file-like response (tests inject one). Raises FetchError listing
    every mirror's failure."""
    opener = opener or _default_open
    errors = []
    for url in urls:
        if cancel and cancel():
            raise FetchError("已取消")
        try:
            with opener(url, timeout) as resp:
                total = int(getattr(resp, "headers", {}).get("Content-Length") or 0) if hasattr(resp, "headers") else 0
                chunks, got = [], 0
                while True:
                    if cancel and cancel():
                        raise FetchError("已取消")
                    chunk = resp.read(64 * 1024)
                    if not chunk:
                        break
                    got += len(chunk)
                    if got > max_bytes:
                        raise FetchError(f"文件过大（>{max_bytes // 1024} KB）")
                    chunks.append(chunk)
                    if progress:
                        progress(got, total)
            data = b"".join(chunks)
            if validate:
                validate(data)
            return data, url
        except FetchError as exc:
            if str(exc) == "已取消":
                raise
            errors.append(f"{_host(url)}：{exc}")
        except UpdateError as exc:
            errors.append(f"{_host(url)}：{exc}")
        except Exception as exc:   # URLError, timeouts, ssl, connection resets …
            errors.append(f"{_host(url)}：{_short_error(exc)}")
    raise _summarize(errors, len(urls))


def _summarize(errors, tried) -> "FetchError":
    """One short Chinese line for the panel; every mirror's reason stays in ``.details`` (and the console)."""
    print("[Wondful] 更新：" + "；".join(errors))
    bad = next((e.split("：", 1)[1] for e in errors if "校验失败" in e or "格式错误" in e or "无效" in e), "")
    if bad:
        exc = FetchError(bad if bad.startswith("校验失败") else "更新信息无效：" + bad)
    else:
        exc = FetchError(f"无法连接更新服务器（已尝试 {tried} 个地址）")
    exc.details = list(errors)
    return exc


def _default_open(url, timeout):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Cache-Control": "no-cache"})
    return urllib.request.urlopen(req, timeout=timeout)


def fetch_manifest(*, opener=None, bases=None, timeout=MANIFEST_TIMEOUT) -> dict:
    holder = {}

    def _validate(data):
        holder["m"] = parse_manifest(data)

    _data, url = fetch_first(mirror_urls(MANIFEST_PATH, bases), timeout=timeout, opener=opener, validate=_validate)
    manifest = holder["m"]
    manifest["source"] = url
    return manifest


def verify_bytes(data: bytes, manifest: dict) -> None:
    if len(data) != int(manifest["size"]):
        raise UpdateError(f"校验失败：文件大小 {len(data)} 字节，应为 {manifest['size']} 字节")
    digest = hashlib.sha256(data).hexdigest()
    if digest != manifest["sha256"]:
        raise UpdateError(f"校验失败：sha256 不一致（{digest[:12]}…，应为 {manifest['sha256'][:12]}…）")


def download_zip(manifest: dict, dest_dir, *, opener=None, bases=None, progress=None, cancel=None,
                 timeout=DOWNLOAD_TIMEOUT) -> Path:
    """Download + verify (a mirror that serves a wrong file falls through to the next). Returns the zip path."""
    data, _url = fetch_first(mirror_urls("releases/" + manifest["zip"], bases), timeout=timeout,
                             max_bytes=MAX_ZIP_BYTES, opener=opener, progress=progress, cancel=cancel,
                             validate=lambda d: verify_bytes(d, manifest))
    path = Path(dest_dir) / manifest["zip"]
    path.write_bytes(data)
    return path


# ---------------------------------------------------------------------------
# throttle state
# ---------------------------------------------------------------------------

def load_state(state_dir) -> dict:
    try:
        obj = json.loads((Path(state_dir) / STATE_FILE).read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except (OSError, ValueError):
        return {}


def save_state(state_dir, state: dict) -> None:
    try:
        path = Path(state_dir)
        path.mkdir(parents=True, exist_ok=True)
        tmp = path / (STATE_FILE + ".tmp")
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, path / STATE_FILE)
    except OSError:
        pass


def should_check(state: dict, now=None, interval=CHECK_INTERVAL) -> bool:
    """At most once per ``interval``; a clock set back (last check in the future) checks again."""
    now = time.time() if now is None else now
    try:
        last = float(state.get("last_check", 0) or 0)
    except (TypeError, ValueError):
        last = 0.0
    return last <= 0 or now - last >= interval or last > now + 60


def record_check(state: dict, manifest, now=None) -> dict:
    state = dict(state)
    state["last_check"] = time.time() if now is None else now
    if manifest:
        state["latest"] = {k: manifest[k] for k in ("version", "zip", "sha256", "size", "notes") if k in manifest}
    return state


def pending_update(state: dict, local_version):
    """The cached manifest when it is newer than ``local_version`` and not ignored; else None."""
    latest = state.get("latest")
    if not isinstance(latest, dict):
        return None
    try:
        manifest = parse_manifest(latest)
    except ManifestError:
        return None
    if not is_newer(manifest["version"], local_version):
        return None
    if is_ignored(state, manifest["version"]):
        return None
    return manifest


def is_ignored(state: dict, version) -> bool:
    """「忽略此版本」 hides exactly that version; a later one prompts again."""
    try:
        return bool(state.get("ignored")) and version_str(state["ignored"]) == version_str(version)
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# install
# ---------------------------------------------------------------------------

def read_bl_info(init_py) -> dict:
    """bl_info of an add-on ``__init__.py`` without importing it; {} when absent/unreadable."""
    try:
        tree = ast.parse(Path(init_py).read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError, ValueError):
        return {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "bl_info" for t in node.targets):
            try:
                info = ast.literal_eval(node.value)
                return info if isinstance(info, dict) else {}
            except ValueError:
                return {}
    return {}


def is_wondful_addon_dir(path) -> bool:
    path = Path(path)
    init = path / "__init__.py"
    if not init.is_file():
        return False
    if path.name == PACKAGE_DIR_NAME:
        return True
    return "wondful ai" in str(read_bl_info(init).get("name", "")).lower()


def safe_extract(zip_path, dest) -> Path:
    """Extract ``zip_path`` into ``dest`` (rejecting absolute/.. paths); return dest/wondful_ai_renderer."""
    dest = Path(dest).resolve()
    try:
        with zipfile.ZipFile(zip_path) as zf:
            for info in zf.infolist():
                name = info.filename.replace("\\", "/")
                target = (dest / name).resolve()
                if name.startswith("/") or ".." in name.split("/") or not str(target).startswith(str(dest)):
                    raise UpdateError(f"安装包含有不安全的路径：{info.filename}")
            zf.extractall(dest)
    except zipfile.BadZipFile:
        raise UpdateError("安装包不是有效的 ZIP 文件") from None
    pkg = dest / PACKAGE_DIR_NAME
    if not (pkg / "__init__.py").is_file():
        raise UpdateError(f"安装包里没有 {PACKAGE_DIR_NAME}/__init__.py")
    return pkg


def validate_package(pkg_dir, expected_version) -> None:
    info = read_bl_info(Path(pkg_dir) / "__init__.py")
    if "wondful ai" not in str(info.get("name", "")).lower():
        raise UpdateError("安装包不是 Wondful AI 渲染器")
    try:
        got = version_str(info.get("version"))
    except ValueError:
        raise UpdateError("安装包里的版本号无效") from None
    if got != version_str(expected_version):
        raise UpdateError(f"安装包版本 {got} 与更新信息 {version_str(expected_version)} 不一致")


def cleanup_leftovers(parent) -> list:
    """Remove staging/backup dirs an interrupted or locked update left beside the add-on."""
    removed = []
    try:
        for child in Path(parent).iterdir():
            if child.is_dir() and child.name.startswith((STAGING_PREFIX, BACKUP_PREFIX)):
                shutil.rmtree(child, ignore_errors=True)
                if not child.exists():
                    removed.append(str(child))
    except OSError:
        pass
    return removed


def install_zip(zip_path, target_dir, expected_version, *, phase=None) -> dict:
    """Extract → validate → swap into ``target_dir`` (the loaded add-on dir). On any failure before the swap
    completes, ``target_dir`` is exactly as before. Returns {"backup_removed": bool, "backup": path}."""
    target = Path(target_dir).resolve()
    parent = target.parent
    if not target.is_dir():
        raise UpdateError(f"找不到当前插件目录：{target}")
    if not os.access(parent, os.W_OK):
        raise UpdateError(f"插件目录没有写入权限：{parent}")
    stamp = time.strftime("%Y%m%d%H%M%S")
    staging = Path(tempfile.mkdtemp(prefix=STAGING_PREFIX, dir=parent))
    backup = parent / f"{BACKUP_PREFIX}{stamp}_{os.getpid()}"
    try:
        if phase:
            phase("安装：解压")
        pkg = safe_extract(zip_path, staging / "x")
        validate_package(pkg, expected_version)
        if phase:
            phase("安装：替换文件")
        os.replace(target, backup)              # same filesystem: atomic rename
        try:
            os.replace(pkg, target)
        except Exception:
            if not target.exists():
                os.replace(backup, target)      # roll back
            raise
    except UpdateError:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    except Exception as exc:
        shutil.rmtree(staging, ignore_errors=True)
        raise UpdateError(f"安装失败：{exc}") from None
    shutil.rmtree(staging, ignore_errors=True)
    shutil.rmtree(backup, ignore_errors=True)   # loaded .pyc on Windows may stay locked: cleaned next start
    return {"backup": str(backup), "backup_removed": not backup.exists()}


# ---------------------------------------------------------------------------
# duplicate installs
# ---------------------------------------------------------------------------

def find_duplicates(current_dir, roots) -> list:
    """Other Wondful add-on copies Blender would see (non-hidden child dirs of each add-on/extension root)."""
    try:
        current = Path(current_dir).resolve()
    except OSError:
        current = Path(current_dir)
    found, seen = [], set()
    for root in roots or ():
        try:
            children = sorted(Path(root).iterdir())
        except OSError:
            continue
        for child in children:
            if child.name.startswith(".") or not child.is_dir():
                continue
            try:
                resolved = child.resolve()
            except OSError:
                continue
            if resolved == current or str(resolved) in seen:
                continue
            if is_wondful_addon_dir(child):
                seen.add(str(resolved))
                found.append(child)
    return found


def remove_dirs(paths) -> tuple:
    """rmtree each (symlinks only unlinked); returns (removed, failed[(path, reason)])."""
    removed, failed = [], []
    for path in paths:
        path = Path(path)
        try:
            if path.is_symlink():
                path.unlink()
            else:
                shutil.rmtree(path)
            removed.append(str(path))
        except OSError as exc:
            failed.append((str(path), str(exc)[:120]))
    return removed, failed
