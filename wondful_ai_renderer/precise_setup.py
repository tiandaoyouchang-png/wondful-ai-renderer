"""1.2.5 ④ 场景粗模 ·「精确模式」— first-use install of the local depth model (no bpy, unit-tested).

精确模式 measures the reference photo with MoGe-2 ViT-S (Microsoft, MIT licence; ONNX export
``Ruicheng/moge-2-vits-normal-onnx``) on the CPU, so the photo never leaves the computer for this step.
It needs two things the add-on does not ship:

* the model file ``moge2_vits_normal.onnx`` (140 852 051 bytes, sha256 pinned below);
* ``onnxruntime`` 1.20.1 (CPU wheel from PyPI, ~11-31 MB) for the running Python (Blender 4.x / 5.x = 3.11;
  3.10 / 3.12 / 3.13 wheels are listed too) on Windows x64, macOS (universal2, macOS 13+), Linux x86_64 / aarch64.

Both are downloaded once, verified (size + sha256; a mirror that serves a wrong file falls through to the
next), and kept in the add-on's per-user config folder ``~/.wondful_ai_renderer/precise`` (Windows:
``C:\\Users\\<name>\\.wondful_ai_renderer\\precise``) — OUTSIDE the add-on folder, so 一键更新 never wipes them.
``WONDFUL_PRECISE_DIR`` overrides (tests).

The wheel is a zip: it is extracted (no pip) into ``<dir>/site/ort-1.20.1-<pytag>-<plat>/``, which is put on
``sys.path`` only when 精确模式 runs. An onnxruntime that is already importable (another add-on) is used as is.
onnxruntime's declared extras (coloredlogs, flatbuffers, packaging, protobuf, sympy) are not needed for
``InferenceSession`` on CPU, so they are not installed. Every failure raises ``PreciseInstallError`` with one
Chinese line; the caller then falls back to the normal (LLM-only) analysis.
"""
from __future__ import annotations

import hashlib
import os
import platform
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

ORT_VERSION = "1.20.1"
MODEL_FILE = "moge2_vits_normal.onnx"
MODEL_SHA256 = "24eacb5dc7a2c54c7bc98f7de085ffbed79ad006ea5b664c2c2cdc02ff3a52f0"
MODEL_SIZE = 140852051
MODEL_URLS = (
    "https://huggingface.co/Ruicheng/moge-2-vits-normal-onnx/resolve/main/model.onnx",
    "https://hf-mirror.com/Ruicheng/moge-2-vits-normal-onnx/resolve/main/model.onnx",
)
# PyPI file host first, then China-reachable PyPI mirrors (same /packages/... layout)
WHEEL_BASES = (
    "https://files.pythonhosted.org",
    "https://pypi.tuna.tsinghua.edu.cn",
    "https://mirrors.aliyun.com/pypi",
)
# onnxruntime 1.20.1 CPU wheels: "<pytag>-<plat>": (path on the PyPI file host, sha256, size)
WHEELS = {
    "cp310-linux_aarch64": ("/packages/6d/c6/c4c0860bee2fde6037bdd9dcd12d323f6e38cf00fcc9a5065b394337fc55/onnxruntime-1.20.1-cp310-cp310-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl",
                            "7b2908b50101a19e99c4d4e97ebb9905561daf61829403061c1adc1b588bc0de", 11954010),
    "cp310-linux_x86_64": ("/packages/63/47/3dc0b075ab539f16b3d8b09df6b504f51836086ee709690a6278d791737d/onnxruntime-1.20.1-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl",
                           "d82daaec24045a2e87598b8ac2b417b1cce623244e80e663882e9fe1aae86410", 13330452),
    "cp310-macos": ("/packages/4e/28/99f903b0eb1cd6f3faa0e343217d9fb9f47b84bca98bd9859884631336ee/onnxruntime-1.20.1-cp310-cp310-macosx_13_0_universal2.whl",
                    "e50ba5ff7fed4f7d9253a6baf801ca2883cc08491f9d32d78a80da57256a5439", 30996314),
    "cp310-win_amd64": ("/packages/a9/e6/33ab10066c9875a29d55e66ae97c3bf91b9b9b987179455d67c32261a49c/onnxruntime-1.20.1-cp310-cp310-win_amd64.whl",
                        "d3b616bb53a77a9463707bb313637223380fc327f5064c9a782e8ec69c22e6a2", 11329702),
    "cp311-linux_aarch64": ("/packages/a5/da/c44bf9bd66cd6d9018a921f053f28d819445c4d84b4dd4777271b0fe52a2/onnxruntime-1.20.1-cp311-cp311-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl",
                            "f6243e34d74423bdd1edf0ae9596dd61023b260f546ee17d701723915f06a9f7", 11955227),
    "cp311-linux_x86_64": ("/packages/11/ac/4120dfb74c8e45cce1c664fc7f7ce010edd587ba67ac41489f7432eb9381/onnxruntime-1.20.1-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl",
                           "5eec64c0269dcdb8d9a9a53dc4d64f87b9e0c19801d9321246a53b7eb5a7d1bc", 13331703),
    "cp311-macos": ("/packages/95/8d/2634e2959b34aa8a0037989f4229e9abcfa484e9c228f99633b3241768a6/onnxruntime-1.20.1-cp311-cp311-macosx_13_0_universal2.whl",
                    "06bfbf02ca9ab5f28946e0f912a562a5f005301d0c419283dc57b3ed7969bb7b", 30998725),
    "cp311-win_amd64": ("/packages/2c/2d/2d4d202c0bcfb3a4cc2b171abb9328672d7f91d7af9ea52572722c6d8d96/onnxruntime-1.20.1-cp311-cp311-win_amd64.whl",
                        "8508887eb1c5f9537a4071768723ec7c30c28eb2518a00d0adcd32c89dea3221", 11329895),
    "cp312-linux_aarch64": ("/packages/c5/9d/a42a84e10f1744dd27c6f2f9280cc3fb98f869dd19b7cd042e391ee2ab61/onnxruntime-1.20.1-cp312-cp312-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl",
                            "f1f56e898815963d6dc4ee1c35fc6c36506466eff6d16f3cb9848cea4e8c8172", 11952833),
    "cp312-linux_x86_64": ("/packages/47/42/2f71f5680834688a9c81becbe5c5bb996fd33eaed5c66ae0606c3b1d6a02/onnxruntime-1.20.1-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl",
                           "bb71a814f66517a65628c9e4a2bb530a6edd2cd5d87ffa0af0f6f773a027d99e", 13333903),
    "cp312-macos": ("/packages/e5/39/9335e0874f68f7d27103cbffc0e235e32e26759202df6085716375c078bb/onnxruntime-1.20.1-cp312-cp312-macosx_13_0_universal2.whl",
                    "22b0655e2bf4f2161d52706e31f517a0e54939dc393e92577df51808a7edc8c9", 31007580),
    "cp312-win_amd64": ("/packages/dd/80/76979e0b744307d488c79e41051117634b956612cc731f1028eb17ee7294/onnxruntime-1.20.1-cp312-cp312-win_amd64.whl",
                        "19c2d843eb074f385e8bbb753a40df780511061a63f9def1b216bf53860223fb", 11331482),
    "cp313-linux_aarch64": ("/packages/81/0d/13bbd9489be2a6944f4a940084bfe388f1100472f38c07080a46fbd4ab96/onnxruntime-1.20.1-cp313-cp313-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl",
                            "fb44b08e017a648924dbe91b82d89b0c105b1adcfe31e90d1dc06b8677ad37be", 11951459),
    "cp313-linux_x86_64": ("/packages/c0/ea/4454ae122874fd52bbb8a961262de81c5f932edeb1b72217f594c700d6ef/onnxruntime-1.20.1-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl",
                           "bda6aebdf7917c1d811f21d41633df00c58aff2bef2f598f69289c1f1dabc4b3", 13331620),
    "cp313-macos": ("/packages/f7/71/c5d980ac4189589267a06f758bd6c5667d07e55656bed6c6c0580733ad07/onnxruntime-1.20.1-cp313-cp313-macosx_13_0_universal2.whl",
                    "cc01437a32d0042b606f462245c8bbae269e5442797f6213e36ce61d5abdd8cc", 31007574),
    "cp313-win_amd64": ("/packages/d8/e0/50db43188ca1c945decaa8fc2a024c33446d31afed40149897d4f9de505f/onnxruntime-1.20.1-cp313-cp313-win_amd64.whl",
                        "d30367df7e70f1d9fc5a6a68106f5961686d39b54d3221f760085524e8d38e16", 11331758),
}
TIMEOUT = 20.0                  # socket timeout per mirror (not total)
CHUNK = 256 * 1024
USER_AGENT = "WondfulAIRenderer-Precise"
APPROX_MB = 155                 # shown to the user before the first download


class PreciseInstallError(Exception):
    """One Chinese line for the panel; ``details`` keeps every mirror's reason."""

    def __init__(self, message, details=None, cancelled=False):
        super().__init__(message)
        self.details = list(details or [])
        self.cancelled = cancelled


# ---------------------------------------------------------------------------------------- paths

def data_dir(env=None, home=None) -> Path:
    """The add-on's per-user config folder (``~/.wondful_ai_renderer``, same as timings / update state) + ``precise``.
    Outside every add-on folder, so 一键更新 / reinstall never wipes the 155 MB download."""
    env = os.environ if env is None else env
    over = (env.get("WONDFUL_PRECISE_DIR") or "").strip()
    if over:
        return Path(over)
    return (Path(home) if home else Path.home()) / ".wondful_ai_renderer" / "precise"


def platform_key(system=None, machine=None, version=None):
    """``cp311-win_amd64`` etc. for this Python, or None when no wheel is listed."""
    system = system or platform.system()
    machine = (machine or platform.machine() or "").lower()
    v = version or sys.version_info
    py = f"cp{v[0]}{v[1]}"
    if system == "Windows":
        plat = "win_amd64" if machine in ("amd64", "x86_64") else None
    elif system == "Darwin":
        plat = "macos"
    elif system == "Linux":
        plat = {"x86_64": "linux_x86_64", "amd64": "linux_x86_64", "aarch64": "linux_aarch64",
                "arm64": "linux_aarch64"}.get(machine)
    else:
        plat = None
    key = f"{py}-{plat}" if plat else None
    return key if key in WHEELS else None


def model_path(root=None) -> Path:
    return Path(root or data_dir()) / MODEL_FILE


def site_dir(key, root=None) -> Path:
    return Path(root or data_dir()) / "site" / f"ort-{ORT_VERSION}-{key}"


def wheel_urls(key) -> list:
    path = WHEELS[key][0]
    return [b + path for b in WHEEL_BASES]


# ---------------------------------------------------------------------------------------- state

def model_ok(root=None, deep=False) -> bool:
    p = model_path(root)
    if not p.is_file() or p.stat().st_size != MODEL_SIZE:
        return False
    return file_sha256(p) == MODEL_SHA256 if deep else True


def ort_installed(key, root=None) -> bool:
    return key is not None and (site_dir(key, root) / "onnxruntime" / "__init__.py").is_file() and \
        (site_dir(key, root) / ".complete").is_file()


def status(root=None, key=None) -> dict:
    """{"ready": bool, "model": bool, "runtime": "system"|"local"|"", "key": ..., "dir": ...} (cheap; no import)."""
    key = key if key is not None else platform_key()
    rt = ""
    if _ort_already_importable():
        rt = "system"
    elif ort_installed(key, root):
        rt = "local"
    m = model_ok(root)
    return {"ready": bool(m and rt), "model": m, "runtime": rt, "key": key, "dir": str(Path(root or data_dir()))}


_CACHE = {"t": 0.0, "v": None}


def cached_status(ttl=5.0) -> dict:
    """status() for the panel draw (re-checked at most every ``ttl`` seconds)."""
    import time
    now = time.time()
    if _CACHE["v"] is None or now - _CACHE["t"] > ttl:
        try:
            _CACHE["v"] = status()
        except Exception:
            _CACHE["v"] = {"ready": False, "model": False, "runtime": "", "key": None, "dir": ""}
        _CACHE["t"] = now
    return _CACHE["v"]


def _ort_already_importable() -> bool:
    if "onnxruntime" in sys.modules:
        return True
    try:
        import importlib.util
        return importlib.util.find_spec("onnxruntime") is not None
    except Exception:
        return False


def import_onnxruntime(root=None, key=None):
    """An importable onnxruntime: the one already present, else the plugin-local site dir."""
    try:
        import onnxruntime as ort  # noqa: F401
        return ort
    except Exception:
        pass
    key = key if key is not None else platform_key()
    if key is None:
        raise PreciseInstallError(f"精确模式不支持这台电脑（{platform.system()} {platform.machine()}，Python "
                                  f"{sys.version_info[0]}.{sys.version_info[1]}）")
    sd = site_dir(key, root)
    if not ort_installed(key, root):
        raise PreciseInstallError("深度模型运行库还没有安装")
    if str(sd) not in sys.path:
        sys.path.insert(0, str(sd))
    try:
        import onnxruntime as ort
        return ort
    except Exception as exc:
        sys.modules.pop("onnxruntime", None)
        try:
            sys.path.remove(str(sd))
        except ValueError:
            pass
        raise PreciseInstallError(f"深度模型运行库无法加载：{str(exc)[:120]}") from exc


# ---------------------------------------------------------------------------------------- download

def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _default_open(url, timeout):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    return urllib.request.urlopen(req, timeout=timeout)


def _host(url) -> str:
    return url.split("//", 1)[-1].split("/", 1)[0]


def _short(exc) -> str:
    code = getattr(exc, "code", None)
    if code:
        return f"HTTP {code}"
    text = str(getattr(exc, "reason", None) or exc) or exc.__class__.__name__
    return ("超时" if "timed out" in text.lower() else text)[:80]


def download_verified(urls, dest, sha256, size, *, opener=None, progress=None, cancel=None, timeout=TIMEOUT) -> str:
    """Stream the first mirror that serves the right bytes to ``dest`` (via ``dest.part``; sha256 + size checked).
    ``progress(done_bytes, total_bytes)``; ``cancel()`` → PreciseInstallError(cancelled=True). Returns the URL used."""
    opener = opener or _default_open
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    errors = []
    for url in urls:
        if cancel and cancel():
            raise PreciseInstallError("已取消", cancelled=True)
        h = hashlib.sha256()
        got = 0
        try:
            with opener(url, timeout) as resp, open(part, "wb") as out:
                while True:
                    if cancel and cancel():
                        raise PreciseInstallError("已取消", cancelled=True)
                    chunk = resp.read(CHUNK)
                    if not chunk:
                        break
                    got += len(chunk)
                    if got > size:
                        raise ValueError("文件过大")
                    h.update(chunk)
                    out.write(chunk)
                    if progress:
                        progress(got, size)
            if got != size:
                raise ValueError(f"大小 {got}，应为 {size}")
            if h.hexdigest() != sha256:
                raise ValueError(f"sha256 不一致（{h.hexdigest()[:12]}…）")
            os.replace(part, dest)
            return url
        except PreciseInstallError:
            _rm(part)
            raise
        except ValueError as exc:
            _rm(part)
            errors.append(f"{_host(url)}：校验失败 {exc}")
        except Exception as exc:
            _rm(part)
            errors.append(f"{_host(url)}：{_short(exc)}")
    print("[Wondful] 精确模式下载：" + "；".join(errors))
    bad = [e for e in errors if "校验失败" in e]
    msg = ("下载的文件校验失败" if bad and len(bad) == len(errors) else f"无法下载（已尝试 {len(urls)} 个地址）")
    raise PreciseInstallError(msg, errors)


def _rm(p):
    try:
        Path(p).unlink()
    except OSError:
        pass


def install_wheel(whl, key, root=None) -> Path:
    """Extract the wheel into the version-specific site dir (atomic via a temp dir + ``.complete`` marker)."""
    sd = site_dir(key, root)
    tmp = sd.with_name(sd.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    try:
        with zipfile.ZipFile(whl) as zf:
            for info in zf.infolist():
                name = info.filename
                if name.startswith("/") or ".." in Path(name).parts:
                    raise PreciseInstallError("运行库安装包无效")
            zf.extractall(tmp)
        if not (tmp / "onnxruntime" / "__init__.py").is_file():
            raise PreciseInstallError("运行库安装包无效（缺少 onnxruntime）")
        (tmp / ".complete").write_text(ORT_VERSION, encoding="utf-8")
        shutil.rmtree(sd, ignore_errors=True)
        os.replace(tmp, sd)
    except zipfile.BadZipFile as exc:
        shutil.rmtree(tmp, ignore_errors=True)
        raise PreciseInstallError("运行库安装包无效") from exc
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return sd


def ensure_installed(*, root=None, key=None, opener=None, progress=None, cancel=None, check_import=True) -> dict:
    """Download what is missing (runtime wheel first: small; then the model), then import onnxruntime.
    ``progress(fraction 0..1, text)``. Returns status(). Raises PreciseInstallError."""
    root = Path(root or data_dir())
    key = key if key is not None else platform_key()
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise PreciseInstallError(f"无法创建缓存目录 {root}：{_short(exc)}") from exc
    need_rt = not _ort_already_importable() and not ort_installed(key, root)
    need_model = not model_ok(root)
    if need_rt and key is None:
        raise PreciseInstallError(f"精确模式不支持这台电脑（{platform.system()} {platform.machine()}，Python "
                                  f"{sys.version_info[0]}.{sys.version_info[1]}）")
    total = (WHEELS[key][2] if need_rt else 0) + (MODEL_SIZE if need_model else 0)
    done0 = [0]

    def _p(label):
        def cb(got, _size):
            if progress and total:
                progress(min(1.0, (done0[0] + got) / float(total)),
                         f"{label} {(done0[0] + got) / 1e6:.0f}/{total / 1e6:.0f} MB")
        return cb
    if need_rt:
        path, sha, size = WHEELS[key]
        whl = root / "downloads" / Path(path).name
        try:
            download_verified(wheel_urls(key), whl, sha, size, opener=opener, progress=_p("下载运行库"), cancel=cancel)
            install_wheel(whl, key, root)
        finally:
            _rm(whl)
        done0[0] += size
    if need_model:
        download_verified(MODEL_URLS, model_path(root), MODEL_SHA256, MODEL_SIZE, opener=opener,
                          progress=_p("下载深度模型"), cancel=cancel)
    if check_import:
        import_onnxruntime(root, key)
    return status(root, key)
