"""3.2.1 · Codex 图像编辑（直连）.

Calls the hosted ``image_generation`` tool of the Codex Responses backend
directly, on the user's existing ChatGPT/Codex login (``~/.codex/auth.json``),
so the Blender camera base is *edited* (image 1 = canvas, optional
``input_image_mask``) instead of an agent re-drawing a new picture from
references.  Pure standard library + numpy; safe on a worker thread (no bpy).

Payload, headers, SSE parsing and token refresh follow the open-source
reference clients jdmnk/codex-imagegen-cli and the PyPI package
codex-image-gen.  Any failure raises :class:`DirectEditError`; the caller
falls back to the ``codex exec`` agent path.  Tokens are never logged.
"""
from __future__ import annotations

import base64
import binascii
import json
import math
import os
import platform
import re
import struct
import threading
import time
import uuid
import zlib
from pathlib import Path
from urllib import error as urlerror
from urllib import request as urlrequest

DEFAULT_BASE_URL = "https://chatgpt.com/backend-api/codex"
DEFAULT_REFRESH_URL = "https://auth.openai.com/oauth/token"
CODEX_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
DEFAULT_MODEL = "gpt-6.1-sol"
# Tried in order when the account rejects a model (HTTP 400 "not supported").
FALLBACK_MODELS = ("gpt-6.1-sol", "gpt-6-sol", "gpt-5.6-sol", "gpt-5.6-terra")
IMAGE_MODEL = "gpt-image-2"
ORIGINATOR = "codex_cli_rs"
MAX_EDIT_IMAGES = 5
MAX_IMAGE_BYTES = 32 * 1024 * 1024
ADDON_UA = "wondful-ai-renderer/1.1.1"
BASE_URL_ENV = "WONDFUL_CODEX_IMAGE_BASE_URL"
REFRESH_URL_ENV = "WONDFUL_CODEX_REFRESH_URL"

INSTRUCTIONS = (
    "Use the image_generation tool exactly once to EDIT the first input image and return one PNG. "
    "The first input image is the canvas: keep its framing, camera, product geometry, position, scale, "
    "perspective and silhouette. Later input images are references only. Do not use any other tool."
)
# 1.2.0 草图模式: the clay product is a rough visual hull built from the user's three-view images.
SKETCH_INSTRUCTIONS = (
    "Use the image_generation tool exactly once to EDIT the first input image and return one PNG. "
    "The first input image is the canvas: keep its framing, camera, horizon and the product's position, "
    "orientation, scale and perspective. The gray product in it is only a rough proxy built from the "
    "product's three-view drawings (the later input images): take the product's exact shape details, "
    "materials, colours and logos from those three views and correct the proxy's blocky faces and gaps. "
    "Never copy the three views' layout, background or lighting into the result. Do not use any other tool."
)
SKETCH_REF_LIMIT = 3

ROLE_LABELS = {
    "camera": "Blender 相机白模底图（构图 / 几何的唯一权威）",
    "structure": "Blender 结构控制图（只管几何，不是外观）",
    "product": "产品三视图（只看产品造型与身份细节，不取其打光和背景）",
    "person": "人物参考（只看人物身份）",
    "environment": "环境参考图（决定环境、光线方向、色温与氛围）",
    "appearance": "外观参考图集（格内标签见图上 R 编号）",
    "control_bundle": "参考图集（格内标签见图上 R 编号）",
}
# Lower number = kept first when the 5-image limit forces a drop.
SIMPLE_ENV_LIMIT = 2
ROLE_PRIORITY = {"camera": 0, "product": 1, "environment": 2, "appearance": 2, "control_bundle": 2,
                 "person": 3, "structure": 4}


class DirectEditError(RuntimeError):
    """Direct edit failed; ``fatal`` = do not retry direct for the rest of this render."""

    def __init__(self, message, *, kind="error", status=0, fatal=False, detail=""):
        super().__init__(message)
        self.detail = detail
        self.kind = kind
        self.status = int(status or 0)
        self.fatal = bool(fatal)


# --------------------------------------------------------------------------- auth

def auth_file_candidates() -> list[Path]:
    paths = []
    home_env = os.environ.get("CODEX_HOME", "").strip()
    if home_env:
        paths.append(Path(home_env).expanduser() / "auth.json")
    userprofile = os.environ.get("USERPROFILE", "").strip()
    if userprofile:
        paths.append(Path(userprofile) / ".codex" / "auth.json")
    paths.append(Path.home() / ".codex" / "auth.json")
    seen, out = set(), []
    for p in paths:
        key = str(p)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def find_auth_file() -> Path | None:
    for p in auth_file_candidates():
        try:
            if p.is_file():
                return p
        except OSError:
            continue
    return None


def _b64url_json(segment: str) -> dict:
    try:
        raw = base64.urlsafe_b64decode(segment + "=" * (-len(segment) % 4))
        data = json.loads(raw.decode("utf-8"))
        return data if isinstance(data, dict) else {}
    except (binascii.Error, ValueError, UnicodeDecodeError):
        return {}


def jwt_claims(token) -> dict:
    if not isinstance(token, str) or token.count(".") < 2:
        return {}
    return _b64url_json(token.split(".")[1])


def _id_claims(tokens: dict) -> dict:
    id_token = tokens.get("id_token")
    if isinstance(id_token, dict):  # legacy layout
        if isinstance(id_token.get("raw_jwt"), str):
            return (jwt_claims(id_token["raw_jwt"]).get("https://api.openai.com/auth") or {})
        return id_token
    return jwt_claims(id_token).get("https://api.openai.com/auth") or {}


def load_auth(path: Path) -> dict:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DirectEditError("未找到 Codex 登录文件 auth.json", kind="auth", fatal=True) from exc
    except (OSError, ValueError) as exc:
        raise DirectEditError("Codex 登录文件 auth.json 无法读取", kind="auth", fatal=True) from exc
    if not isinstance(data, dict) or not isinstance(data.get("tokens"), dict):
        raise DirectEditError("auth.json 里没有 ChatGPT 登录令牌（可能是 API Key 登录或系统钥匙串存储）",
                              kind="auth", fatal=True)
    return data


def account_id(auth: dict) -> str:
    tokens = auth.get("tokens") or {}
    value = tokens.get("account_id")
    if isinstance(value, str) and value:
        return value
    value = _id_claims(tokens).get("chatgpt_account_id")
    return value if isinstance(value, str) else ""


def access_token(auth: dict) -> str:
    value = (auth.get("tokens") or {}).get("access_token")
    return value if isinstance(value, str) else ""


def token_expiring(token: str, leeway: int = 120) -> bool:
    exp = jwt_claims(token).get("exp")
    return isinstance(exp, (int, float)) and exp <= time.time() + leeway


_REFRESH_LOCK = threading.Lock()


def refresh_auth(auth: dict, path: Path, timeout: float = 30) -> dict:
    """Refresh with the stored refresh_token and write auth.json back atomically.

    Re-reads the file first: if Codex already refreshed it, the newer tokens are
    used instead of rotating the refresh token a second time.
    """
    with _REFRESH_LOCK:
        current = load_auth(path)
        if account_id(current) and account_id(auth) and account_id(current) != account_id(auth):
            raise DirectEditError("Codex 账号在生成期间发生变化", kind="auth", fatal=True)
        if current.get("tokens") != auth.get("tokens") and access_token(current) and not token_expiring(access_token(current)):
            return current
        tokens = dict(current.get("tokens") or {})
        refresh_token = tokens.get("refresh_token")
        if not isinstance(refresh_token, str) or not refresh_token:
            raise DirectEditError("Codex 登录已过期且没有 refresh_token，请重新登录 Codex", kind="auth", fatal=True)
        body = json.dumps({"client_id": CODEX_CLIENT_ID, "grant_type": "refresh_token",
                           "refresh_token": refresh_token, "scope": "openid profile email offline_access"}).encode()
        url = os.environ.get(REFRESH_URL_ENV, DEFAULT_REFRESH_URL)
        req = urlrequest.Request(url, data=body, method="POST",
                                 headers={"Content-Type": "application/json", "Accept": "application/json"})
        try:
            with urlrequest.urlopen(req, timeout=timeout) as resp:
                refreshed = json.loads(resp.read().decode("utf-8"))
        except urlerror.HTTPError as exc:
            # OAuth error bodies may echo credentials: never include them.
            raise DirectEditError(f"Codex 令牌刷新失败（HTTP {exc.code}），请在终端重新运行 codex login",
                                  kind="auth", status=exc.code, fatal=True) from exc
        except (urlerror.URLError, OSError, ValueError) as exc:
            raise DirectEditError(f"Codex 令牌刷新失败：{type(exc).__name__}", kind="network", fatal=True) from exc
        if not isinstance(refreshed, dict) or not isinstance(refreshed.get("access_token"), str):
            raise DirectEditError("Codex 令牌刷新没有返回 access_token", kind="auth", fatal=True)
        for key in ("access_token", "refresh_token", "id_token"):
            if isinstance(refreshed.get(key), str) and refreshed[key]:
                tokens[key] = refreshed[key]
        updated = dict(current)
        updated["tokens"] = tokens
        updated["last_refresh"] = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
        if load_auth(path) != current:
            raise DirectEditError("auth.json 在刷新期间被 Codex 修改，放弃写回", kind="auth", fatal=True)
        tmp = Path(str(path) + f".wondful-{os.getpid()}.tmp")
        tmp.write_text(json.dumps(updated, indent=2) + "\n", encoding="utf-8")
        try:
            os.chmod(tmp, 0o600)
        except OSError:
            pass
        os.replace(tmp, path)
        return updated


def _client_version(text: str) -> str:
    m = re.search(r"(\d+\.\d+\.\d+)", text or "")
    return m.group(1) if m else ""


def auth_headers(auth: dict, client_version: str = "") -> dict:
    token = access_token(auth)
    acct = account_id(auth)
    if not token:
        raise DirectEditError("auth.json 里没有 access_token，请重新登录 Codex", kind="auth", fatal=True)
    if not acct:
        raise DirectEditError("auth.json 里没有 ChatGPT 账号 ID，请重新登录 Codex", kind="auth", fatal=True)
    version = _client_version(client_version)
    ua_version = version or "0"
    headers = {
        "Authorization": f"Bearer {token}",
        "ChatGPT-Account-ID": acct,
        "originator": ORIGINATOR,
        "OpenAI-Beta": "responses=experimental",
        "User-Agent": f"{ORIGINATOR}/{ua_version} ({platform.system() or 'unknown'} {platform.release() or ''}; "
                      f"{platform.machine() or 'unknown'}) {ADDON_UA}",
        "session_id": str(uuid.uuid4()),
        "Accept": "text/event-stream",
        "Content-Type": "application/json",
    }
    if version:
        headers["version"] = version
    if _id_claims(auth.get("tokens") or {}).get("chatgpt_account_is_fedramp"):
        headers["X-OpenAI-Fedramp"] = "true"
    return headers


def redact_headers(headers: dict) -> dict:
    out = dict(headers)
    if "Authorization" in out:
        out["Authorization"] = "Bearer <redacted>"
    if out.get("ChatGPT-Account-ID"):
        out["ChatGPT-Account-ID"] = "<redacted>"
    return out


# --------------------------------------------------------------------------- images

def sniff_mime(data: bytes) -> str:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    return "application/octet-stream"


def data_url(path) -> tuple[str, str, int]:
    data = Path(path).read_bytes()
    if len(data) > MAX_IMAGE_BYTES:
        raise DirectEditError(f"图片过大（{len(data) // (1024 * 1024)} MB）：{Path(path).name}", kind="input")
    mime = sniff_mime(data)
    if mime == "application/octet-stream":
        raise DirectEditError(f"无法识别的图片格式：{Path(path).name}", kind="input")
    return f"data:{mime};base64," + base64.b64encode(data).decode("ascii"), mime, len(data)


def choose_edit_size(width: int, height: int, long_edge: int = 2048) -> str:
    """WxH closest to the camera aspect within gpt-image-2 limits.

    Limits (from the reference clients): multiples of 16, edges <= 3840,
    aspect between 1:3 and 3:1, 655,360 - 8,294,400 pixels.
    """
    width, height = max(1, int(width)), max(1, int(height))
    ratio = min(3.0, max(1 / 3.0, width / height))
    target_long = min(3840, max(1024, int(long_edge)))
    best = None
    for long_side in range(max(512, target_long - 256), min(3840, target_long + 256) + 1, 16):
        if ratio >= 1:
            w, h = long_side, max(16, int(round(long_side / ratio / 16.0)) * 16)
        else:
            h, w = long_side, max(16, int(round(long_side * ratio / 16.0)) * 16)
        if max(w, h) > 3840 or max(w, h) > 3 * min(w, h) or not (655360 <= w * h <= 8294400):
            continue
        err = abs((w / h) / ratio - 1.0)
        key = (round(err, 4), abs(long_side - target_long))
        if best is None or key < best[0]:
            best = (key, w, h)
    if best is None:
        return "1536x1024" if ratio >= 1 else "1024x1536"
    return f"{best[1]}x{best[2]}"


def _image_size(path):
    try:
        from .composition import image_dimensions
        size = image_dimensions(path)
        return list(size) if size else []
    except Exception:
        return []


def _png_size(path):
    try:
        with open(path, "rb") as f:
            head = f.read(24)
        if head.startswith(b"\x89PNG\r\n\x1a\n") and head[12:16] == b"IHDR":
            return struct.unpack(">II", head[16:24])
    except OSError:
        pass
    return None


def read_png_rgba(path):
    """Minimal 8-bit, non-interlaced PNG decoder (for our own masks). Returns HxWx4 uint8."""
    import numpy as np
    data = Path(path).read_bytes()
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("not a PNG")
    pos, idat, ihdr = 8, [], None
    while pos + 8 <= len(data):
        length, kind = struct.unpack(">I4s", data[pos:pos + 8])
        chunk = data[pos + 8:pos + 8 + length]
        if kind == b"IHDR":
            ihdr = struct.unpack(">IIBBBBB", chunk)
        elif kind == b"IDAT":
            idat.append(chunk)
        elif kind == b"IEND":
            break
        pos += 12 + length
    if ihdr is None:
        raise ValueError("no IHDR")
    w, h, depth, ctype, _c, _f, interlace = ihdr
    channels = {0: 1, 2: 3, 4: 2, 6: 4}.get(ctype)
    if depth != 8 or interlace or channels is None:
        raise ValueError("unsupported PNG layout")
    raw = zlib.decompress(b"".join(idat))
    stride = w * channels
    rows = np.frombuffer(raw, dtype=np.uint8).reshape(h, stride + 1)
    out = np.zeros((h, stride), dtype=np.uint8)
    prev = np.zeros(stride, dtype=np.int32)
    for y in range(h):
        ftype, line = int(rows[y, 0]), rows[y, 1:].astype(np.int32)
        if ftype == 0:
            cur = line
        elif ftype == 1:
            cur = line.reshape(w, channels).cumsum(axis=0).reshape(-1) & 0xFF
        elif ftype == 2:
            cur = (line + prev) & 0xFF
        else:
            cur = np.zeros(stride, dtype=np.int32)
            for i in range(stride):
                a = int(cur[i - channels]) if i >= channels else 0
                b = int(prev[i])
                c = int(prev[i - channels]) if i >= channels else 0
                if ftype == 3:
                    pred = (a + b) // 2
                else:
                    p = a + b - c
                    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                    pred = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                cur[i] = (int(line[i]) + pred) & 0xFF
        out[y] = cur
        prev = cur
    img = out.reshape(h, w, channels)
    if channels == 4:
        return img
    rgba = np.full((h, w, 4), 255, dtype=np.uint8)
    if channels == 3:
        rgba[:, :, :3] = img
    elif channels == 2:
        rgba[:, :, :3] = img[:, :, :1]
        rgba[:, :, 3] = img[:, :, 1]
    else:  # grayscale: white = editable (OpenAI alpha convention: transparent = edit)
        rgba[:, :, :3] = img
        rgba[:, :, 3] = 255 - img[:, :, 0]
    return rgba


def prepare_mask(mask_path, base_size, out_dir) -> tuple[str, str]:
    """Return (path, note). The mask must be an alpha PNG the same size as image 1."""
    if not mask_path or not Path(mask_path).is_file():
        return "", ""
    mask_size = _png_size(mask_path)
    if mask_size is None:
        return "", "Mask 不是 PNG，已忽略"
    if not base_size or tuple(mask_size) == tuple(base_size):
        return str(mask_path), ""
    import numpy as np
    from .identity_preserve import write_rgba_png
    try:
        rgba = read_png_rgba(mask_path)
    except Exception as exc:
        return "", f"Mask 尺寸与底图不一致且无法重采样（{exc}），已忽略"
    bw, bh = int(base_size[0]), int(base_size[1])
    sh, sw = rgba.shape[:2]
    ys = np.minimum(sh - 1, ((np.arange(bh) + 0.5) * sh / bh).astype(np.int64))
    xs = np.minimum(sw - 1, ((np.arange(bw) + 0.5) * sw / bw).astype(np.int64))
    resized = rgba[ys[:, None], xs[None, :]]
    target = Path(out_dir) / (Path(mask_path).stem + f"_{bw}x{bh}.png")
    return write_rgba_png(target, resized), f"Mask 已重采样到 {bw}×{bh}"



# --------------------------------------------------------------------------- models

_MODEL_CACHE = {"at": 0.0, "models": []}
_MODEL_CACHE_TTL = 3600.0
_MODEL_SKIP = ("codex-auto-review",)


def _rank_models(models) -> list[str]:
    """Visible models, newest first (the server's ``priority``; 0 = latest workhorse)."""
    rows = []
    for index, item in enumerate(models or []):
        if not isinstance(item, dict):
            continue
        slug = str(item.get("slug") or "").strip()
        if not slug or slug in _MODEL_SKIP or item.get("visibility", "list") != "list":
            continue
        if item.get("supported_in_api") is False:
            continue
        try:
            prio = float(item.get("priority", 1000))
        except (TypeError, ValueError):
            prio = 1000.0
        rows.append((prio, index, slug))
    return [slug for _p, _i, slug in sorted(rows)]


def available_models(auth=None, auth_path=None, client_version="", timeout=10.0, *, refresh=False) -> list[str]:
    """Models this ChatGPT account can use right now, newest first.

    Live list from the Codex backend (cached 1 h per Blender session); falls back to the
    Codex CLI's own models_cache.json next to auth.json, then to FALLBACK_MODELS."""
    now = time.time()
    if not refresh and _MODEL_CACHE["models"] and now - _MODEL_CACHE["at"] < _MODEL_CACHE_TTL:
        return list(_MODEL_CACHE["models"])
    ranked = []
    if auth:
        base = os.environ.get(BASE_URL_ENV, "").strip() or DEFAULT_BASE_URL
        version = _client_version(client_version) or "0.162.1"
        url = base.rstrip("/") + "/models?client_version=" + version
        try:
            req = urlrequest.Request(url, headers=auth_headers(auth, client_version))
            with urlrequest.urlopen(req, timeout=timeout) as resp:
                ranked = _rank_models(json.loads(resp.read().decode("utf-8", "replace")).get("models"))
        except Exception:
            ranked = []
    if not ranked and auth_path:
        try:
            cache = json.loads((Path(auth_path).parent / "models_cache.json").read_text(encoding="utf-8"))
            ranked = _rank_models(cache.get("models"))
        except (OSError, ValueError, AttributeError):
            ranked = []
    if ranked:
        _MODEL_CACHE.update(at=now, models=list(ranked))
        return ranked
    return list(FALLBACK_MODELS)


def model_chain(user_model="", available=None) -> list[str]:
    """Models to try in order: the user's pick (if any), then the newest available ones."""
    chain = []
    for slug in [(user_model or "").strip()] + list(available or []) + list(FALLBACK_MODELS):
        if slug and slug not in chain:
            chain.append(slug)
    return chain

# --------------------------------------------------------------------------- payload

def _labels_from_manifest(manifest) -> dict:
    """path -> (role, label) using the reference bundle manifest."""
    out = {}
    for group in manifest or []:
        members = group.get("members") or []
        path = str(group.get("path", ""))
        if not path or not members:
            continue
        roles = [str(m.get("role", "")) for m in members]
        if len(members) == 1:
            role = roles[0]
            sub = str(members[0].get("sub_role") or "")
            label = ROLE_LABELS.get(role, role)
            if role == "structure" and sub:
                label = f"Blender 结构控制图 · {sub}（只管几何，不是外观）"
        else:
            role = min(roles, key=lambda r: ROLE_PRIORITY.get(r, 5))
            cells = "、".join(f"R{m.get('reference')}={m.get('role')}" + (f"/{m.get('sub_role')}" if m.get("sub_role") else "")
                             for m in members)
            label = f"参考图集（格内标签：{cells}；逐格读取，不要画出拼版或标签）"
        out[str(Path(path).resolve())] = (role, label)
    return out


def select_images(edit_base, references, manifest=None, *, repair=False, limit=MAX_EDIT_IMAGES):
    """image[0] = edit base, then the references by priority, within ``limit``."""
    labels = _labels_from_manifest(manifest)
    base = str(Path(edit_base).resolve())
    chosen = [(base, "camera" if not repair else "candidate",
               "编辑画布（Blender 相机白模底图）" if not repair else "编辑画布（上一轮候选图：材质与灯光已基本满意）")]
    rest = []
    for order, ref in enumerate(references or []):
        p = str(Path(ref).resolve())
        if p == base or any(p == c[0] for c in chosen) or any(p == r[2] for r in rest):
            continue
        role, label = labels.get(p, ("camera" if order == 0 else "reference", ROLE_LABELS["camera"] if order == 0 else "参考图"))
        rest.append((ROLE_PRIORITY.get(role, 5), order, p, role, label))
    dropped = []
    keep = sorted(rest, key=lambda r: (r[0], r[1]))
    while len(keep) > limit - 1:
        dropped.append(keep.pop())
    keep.sort(key=lambda r: r[1])  # restore bundle order (camera, structure, product, ...)
    chosen += [(p, role, label) for _pri, _o, p, role, label in keep]
    return chosen, [(p, role) for _pri, _o, p, role, _l in dropped]


def build_edit_prompt(prompt, images, *, mask=False, repair=False, correction=""):
    lines = ["【Codex 图像编辑 · 在 Blender 画布上直接编辑】"]
    if repair:
        lines.append("图 1 是上一轮候选图，作为本次编辑画布；它的材质、光影与环境基本满意，只是产品构图有偏差。"
                     "以图 2（Blender 相机白模）为几何与位置的唯一标准，把产品恢复到白模中的位置、尺度、视角与轮廓。")
    else:
        lines.append("图 1 是编辑画布（Blender 当前相机的白模画面），不是参考图。直接在图 1 上编辑，输出同一画布、同一机位、同一构图。")
    lines.append("必须原样保持：产品的几何造型、画面中的位置、尺度、透视、朝向与轮廓剪影；相机机位、焦段观感、地平线与整体构图。"
                 "禁止平移、缩放、裁切、旋转、换机位或重新设计产品造型。")
    lines.append("只允许改变：材质与表面质感、灯光与阴影、反射、环境与背景、摄影质感——依据后面的参考图和要求。")
    if mask:
        lines.append("本次附带编辑 Mask：Mask 透明区域允许修改，不透明区域保持原样（Logo、字标、铭牌/车牌等身份资产，或修复范围之外的画面）。")
    lines.append("各图职责：")
    for index, (_p, _role, label) in enumerate(images, start=1):
        lines.append(f"- 图 {index}：{label}")
    lines.append("参考图只提供外观信息（产品身份、环境光线与氛围），不得把参考图里的构图、机位、背景布局、拼版、分格线或文字画进结果。"
                 "下文中的 Reference N / R N 是插件的逻辑参考编号，与上面的「图 N」不一定相同；Camera Base / Reference 1 指 Blender 白模。")
    if correction:
        lines.append("本轮纠偏：" + correction.strip())
    lines.append("—— 渲染要求 ——")
    lines.append((prompt or "").strip())
    lines.append("只输出一张图。")
    return "\n".join(lines)


def build_simple_prompt(brief, fallback_prompt="", env_count=0, env_text="", terrain_text="", vehicle=False,
                        massing_text="", sketch_views=None):
    """Short edit prompt for the simple path: only the clay canvas is sent.

    The environment reference is read separately and arrives here as text
    (``env_text``), so its camera and layout can never pull the composition.
    ``vehicle`` picks car wording (车身、车轮、车漆、车牌); anything else gets
    generic product wording with the same product-lock rules.
    ``massing_text`` (1.2.0 环境体块) describes the grey-blue context blocks of the clay base.
    """
    from .product_kind import words
    w = words(bool(vehicle))
    scene = (brief or "").strip()
    env_text = (env_text or "").strip()
    subj = w["subj"]
    sketch_views = [str(v) for v in (sketch_views or []) if str(v).strip()]
    if sketch_views:
        views = "、".join(f"图 {i}：{label}" for i, label in enumerate(sketch_views, start=2))
        lines = [f"把图 1（Blender 白模渲染图）直接编辑成一张写实的商业广告摄影照片，{subj}与环境自然融合。"]
        lines.append(f"后面的图是这件{subj}的三视图（{views}），只用来确定{subj}的外形细节、材质、颜色和 Logo，"
                     "不要把三视图的背景、排版、分格、文字或打光画进结果。")
        lines.append(f"白模是根据三视图生成的粗略外形，保持位置、朝向、大小和透视；外形细节、材质、颜色、Logo 以三视图为准，"
                     f"可修正粗模的块面和缺口（补出镂空、圆角、凹槽、细小部件，去掉三视图里没有的多余体块），"
                     f"但{subj}在画面中占据的范围、朝向和接地位置基本不变；"
                     "相机机位、地平线和构图不变，不平移、不缩放、不裁切、不旋转。")
    else:
        lines = [f"把这张 Blender 白模渲染图直接编辑成一张写实的商业广告摄影照片，{subj}与环境自然融合。"]
        lines.append(f"产品与构图严格保持原图：{subj}在画面中的位置、大小、角度、透视、轮廓、{w['parts']}都不变；"
                     "不增减、不改形任何部件，不重新设计造型；"
                     "相机机位、地平线和构图不变，不平移、不缩放、不裁切、不旋转。")
    lines.append(f"地面的坡度、倾斜方向、起伏以及与{w['contact']}的接触关系以原图白模地面为准（可能是斜坡、坡顶、坡道或路沿；"
                 f"山头/坡顶时{subj}停在山的最高处，四周地面向下掉落，{subj}后方不出现更高的地面，远处只露出低于{subj}的远山、天空或城市），"
                 f"新环境的地面材质贴合这块地面铺设，不要把地面改平、改坡度或让{subj}悬空；相机的俯仰和倾斜也保持不变。")
    if (terrain_text or "").strip():
        lines.append((terrain_text or "").strip())
    massing_text = (massing_text or "").strip()
    if massing_text:
        lines.append(massing_text)
    if vehicle:
        lines.append(("原图里的灰色地面、背景和灯光只是占位（灰蓝色体块按上面的要求变成建筑 / 结构，不删除）：" if massing_text
                      else "原图里的灰色地面、背景和灯光只是占位：") + "全部替换为下面描述的环境，并按新环境重新计算车身的光照、高光、反射和投影，"
                     "不要保留白模里的高光和亮度。车漆保持原本的颜色色相（固有色），明暗随新环境光变化。车身要有真实汽车清漆的分量感和复杂光影：锐利的高光带、天空和远景在车身曲面上的清晰倒影、地平线反射线、深色的下部反射和强烈的明暗过渡，轮毂、玻璃和饰条各有不同的反射质感；不要平涂、不要塑料感或玩具模型感。")
    else:
        lines.append(("原图里的灰色地面、背景和灯光只是占位（灰蓝色体块按上面的要求变成建筑 / 结构，不删除）：" if massing_text
                      else "原图里的灰色地面、背景和灯光只是占位：") + "全部替换为下面描述的环境，并按新环境重新计算产品的光照、高光、反射和投影，"
                     "不要保留白模里的高光和亮度。产品各部分保持原本的颜色色相（固有色）和材质类别，明暗随新环境光变化。"
                     "产品表面材质要真实可信、有分量感：按材质各自表现（织物有纤维纹理和柔和漫反射，皮革有细腻褶皱和柔光，"
                     "木材有木纹，金属有清晰高光和环境倒影，塑料和涂层有对应的光泽），明暗过渡自然；不要平涂、不要塑料感或玩具模型感。")
    if sketch_views:  # the clay proxy has no colour: colour / material come from the three views
        lines = [ln.replace("车漆保持原本的颜色色相（固有色），明暗随新环境光变化。", "车漆颜色和材质以三视图为准，明暗随新环境光变化。")
                   .replace("产品各部分保持原本的颜色色相（固有色）和材质类别，明暗随新环境光变化。",
                            "产品各部分的颜色、材质类别以三视图为准，明暗随新环境光变化。")
                 for ln in lines]
    if env_text and not vehicle:
        # a cached / hand-written description may still carry a car paragraph: drop it
        env_text = "\n".join(p for p in env_text.splitlines() if not p.strip().startswith("车身质感")).strip()
    if env_text:
        lines.append("目标环境（来自环境参考图的文字描述，只用于环境、地面、天气、光线方向、色温和氛围）：" + env_text)
        tag = next((t for t in ("车身质感", "表面质感") if t in env_text), "")
        src = "白模（位置、朝向、大小）和三视图（造型细节、颜色、Logo）" if sketch_views else "白模原图"
        if tag and vehicle:
            lines.append(f"描述里的「{tag}」只用来决定我们这台车的漆面光泽、反射、高光和明暗方式：车型、比例、颜色色相、"
                         f"车门状态、灯组、Logo 和车牌全部以{src}为准，不要照搬参考图里的车。")
        elif tag:
            lines.append(f"描述里的「{tag}」只用来决定我们这件产品表面的光泽、反射、高光和明暗方式：产品的造型、比例、颜色色相、"
                         f"部件和 Logo 全部以{src}为准，不要照搬参考图里的任何物体。")
        lines.append("环境严格按上面的描述还原：只出现描述里写到的元素" + ("和灰蓝色体块对应的建筑 / 结构（它们的风格和材质也按描述来）"
                                                         if massing_text else "")
                     + "，不额外添加" + ("其他" if massing_text else "") + "建筑、遗迹、太阳、眩光、云层等描述没写的东西，"
                     "光线方向、太阳高度、色温和反差与描述一致，不要做得比描述更戏剧化。")
    if scene:
        lines.append("场景要求：" + scene + "。")
    elif not env_text and fallback_prompt:
        lines.append("场景要求：" + fallback_prompt.strip()[:400])
    lines.append("地面接触阴影、反射和环境光与新场景一致，照片质感真实自然。只输出一张图。")
    return "\n".join(lines)


def build_payload(prompt, images, *, size, mask_path="", model="", quality="high", compat=False, instructions=""):
    content = [{"type": "input_text", "text": prompt}]
    summary = []
    for index, (path, role, _label) in enumerate(images, start=1):
        url, mime, nbytes = data_url(path)
        content.append({"type": "input_image", "image_url": url})
        summary.append({"index": index, "role": role, "file": Path(path).name, "mime": mime,
                        "bytes": nbytes, "size": _image_size(path)})
    tool = {"type": "image_generation", "output_format": "png", "size": size}
    if not compat:
        tool.update({"model": IMAGE_MODEL, "quality": quality, "background": "auto"})
    if mask_path:
        url, mime, nbytes = data_url(mask_path)
        tool["input_image_mask"] = {"image_url": url}
        summary.append({"mask": Path(mask_path).name, "mime": mime, "bytes": nbytes,
                        "size": list(_png_size(mask_path) or [])})
    payload = {
        "model": (model or "").strip() or DEFAULT_MODEL,
        "instructions": instructions or INSTRUCTIONS,
        "input": [{"type": "message", "role": "user", "content": content}],
        "tools": [tool],
        "tool_choice": {"type": "image_generation"},
        "parallel_tool_calls": False,
        "store": False,
        "stream": True,
        "include": [],
    }
    return payload, summary


def redact_payload(payload):
    def walk(value):
        if isinstance(value, dict):
            return {k: walk(v) for k, v in value.items()}
        if isinstance(value, list):
            return [walk(v) for v in value]
        if isinstance(value, str) and value.startswith("data:") and ";base64," in value:
            head = value.split(",", 1)[0]
            return f"{head},<{len(value)} chars>"
        return value
    return walk(payload)


# --------------------------------------------------------------------------- transport

def parse_sse_events(text_iter):
    """Yield JSON events from an iterable of text lines (SSE or a single JSON body)."""
    data_lines = []
    for raw in text_iter:
        line = raw.rstrip("\r\n")
        if line == "":
            if data_lines:
                yield "\n".join(data_lines)
                data_lines = []
            continue
        if line.startswith(":") or line.startswith("event:") or line.startswith("id:"):
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
        else:
            data_lines.append(line)
    if data_lines:
        yield "\n".join(data_lines)


def extract_image(events):
    """Return (b64, item, response_id) from Responses stream events."""
    last_status, response_id = "", ""
    for data in events:
        if not data or data == "[DONE]":
            continue
        try:
            event = json.loads(data)
        except ValueError:
            continue
        if not isinstance(event, dict):
            continue
        etype = str(event.get("type", ""))
        if etype in {"response.failed", "response.incomplete", "error"}:
            resp = event.get("response") if isinstance(event.get("response"), dict) else {}
            err = resp.get("error") or event.get("error") or resp.get("incomplete_details") or event
            code = str((err or {}).get("code") or (err or {}).get("type") or (err or {}).get("reason") or etype)
            message = str((err or {}).get("message") or "")[:300]
            kind = "quota" if ("usage_limit" in code or "quota" in code or "rate_limit" in code) else "server"
            raise DirectEditError(f"图像编辑失败：{code} {message}".strip(), kind=kind)
        items = []
        if isinstance(event.get("item"), dict):
            items.append(event["item"])
        if etype == "response.completed" or "output" in event:
            resp = event.get("response") if isinstance(event.get("response"), dict) else event
            response_id = str(resp.get("id") or response_id)
            if isinstance(resp.get("output"), list):
                items.extend(i for i in resp["output"] if isinstance(i, dict))
        for item in items:
            if item.get("type") != "image_generation_call":
                continue
            last_status = str(item.get("status") or last_status)
            result = item.get("result")
            if isinstance(result, str) and result and last_status in {"completed", ""}:
                return result, item, response_id
            if isinstance(result, str) and result:
                return result, item, response_id
    raise DirectEditError("响应里没有图像结果" + (f"（最后状态 {last_status}）" if last_status else ""), kind="parse")


def extract_text(events):
    """Return the assistant text from Responses stream events (text-only call)."""
    deltas, final = [], ""
    for data in events:
        if not data or data == "[DONE]":
            continue
        try:
            event = json.loads(data)
        except ValueError:
            continue
        if not isinstance(event, dict):
            continue
        etype = str(event.get("type", ""))
        if etype in {"response.failed", "response.incomplete", "error"}:
            resp = event.get("response") if isinstance(event.get("response"), dict) else {}
            err = resp.get("error") or event.get("error") or event
            raise DirectEditError(f"环境描述失败：{str((err or {}).get('message') or etype)[:200]}", kind="server")
        if etype == "response.output_text.delta":
            deltas.append(str(event.get("delta") or ""))
        elif etype == "response.output_text.done" and event.get("text"):
            final = str(event["text"])
        elif etype == "response.completed":
            resp = event.get("response") or {}
            for item in resp.get("output") or []:
                for part in (item or {}).get("content") or []:
                    if isinstance(part, dict) and part.get("type") == "output_text" and part.get("text"):
                        final = final or str(part["text"])
    text = (final or "".join(deltas)).strip()
    if not text:
        raise DirectEditError("环境描述为空", kind="parse")
    return text


def _line_iter(resp, cancel_check):
    pending = b""
    read = getattr(resp, "read1", None) or resp.read
    while True:
        if cancel_check and cancel_check():
            raise DirectEditError("任务已取消", kind="cancelled", fatal=True)
        chunk = read(65536)
        if not chunk:
            break
        pending += chunk
        while b"\n" in pending:
            line, pending = pending.split(b"\n", 1)
            yield line.decode("utf-8", errors="replace")
    if pending:
        yield pending.decode("utf-8", errors="replace")


def post_responses(url, headers, payload, *, timeout, cancel_check=None, want="image"):
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    req = urlrequest.Request(url, data=body, headers=headers, method="POST")
    holder = {}
    stop = threading.Event()

    def watch():  # cancellation closes the socket so a blocking read returns
        while not stop.wait(0.5):
            if cancel_check and cancel_check() and holder.get("resp") is not None:
                try:
                    holder["resp"].close()
                except Exception:
                    pass
                return

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    try:
        with urlrequest.urlopen(req, timeout=timeout) as resp:
            holder["resp"] = resp
            events = parse_sse_events(_line_iter(resp, cancel_check))
            return extract_text(events) if want == "text" else extract_image(events)
    except urlerror.HTTPError as exc:
        try:
            text = exc.read().decode("utf-8", errors="replace")
        except Exception:
            text = ""
        raise _http_error(exc.code, text) from exc
    except DirectEditError:
        raise
    except (urlerror.URLError, OSError, ValueError) as exc:
        if cancel_check and cancel_check():
            raise DirectEditError("任务已取消", kind="cancelled", fatal=True) from exc
        reason = getattr(exc, "reason", exc)
        raise DirectEditError(f"网络请求失败：{type(exc).__name__} {str(reason)[:120]}", kind="network") from exc
    finally:
        stop.set()


def _http_error(status, text):
    from .cli_transport import safe_diagnostic
    snippet = safe_diagnostic(text or "")[:300]
    lower = (text or "").lower()
    if status in (401, 403):
        return DirectEditError(f"HTTP {status}：登录无效或无权限", kind="auth", status=status, fatal=status == 403)
    if status == 404:
        return DirectEditError("HTTP 404：接口不存在或已变更", kind="endpoint", status=404, fatal=True, detail=snippet)
    if status == 429:
        quota = "usage_limit" in lower or "insufficient_quota" in lower or "out of credits" in lower
        return DirectEditError(f"HTTP 429：{'额度已用完' if quota else '请求过于频繁'}", kind="quota" if quota else "rate_limit",
                               status=429, fatal=quota)
    if 400 <= status < 500:
        return DirectEditError(f"HTTP {status}：请求被拒绝", kind="request", status=status, detail=snippet)
    return DirectEditError(f"HTTP {status}：服务器错误", kind="server", status=status)


# --------------------------------------------------------------------------- environment description

ENV_DESCRIBE_PROMPT = (
    "你是汽车广告摄影的灯光与美术指导。请只阅读这张环境参考图，写一段 150–220 字的中文环境描述（有车时再加一段车身质感），"
    "供另一张图严格按此重建环境。要求精准、克制，只写图中确实可见的内容，不推测、不美化、不补充。依次写清：\n"
    "1. 场景与背景元素：逐项列出可见的山体、建筑、植被、路牌、护栏等，并写明数量级和远近；\n"
    "2. 地面/路面：材质、颜色、干湿、积雪或积水情况；\n"
    "3. 天气与天空：晴/阴/雾/雪，云量，天空颜色；\n"
    "4. 光线：主光方向（画面左/右/前/后/顶）、太阳高度、太阳是否出现在画面内、光质软硬、阴影长短；\n"
    "5. 色温与主色调、反差、大气通透度与氛围。\n"
    "6. 若图中有车：另起一段以「车身质感：」开头，写 60–120 字，只写材质与光影——车漆光泽和清漆反射强弱、"
    "高光的位置和形状（如逆光轮廓光、顶部高光带、边缘亮线）、天空和环境在车身上的倒影、暗部深浅和明暗过渡、"
    "玻璃、轮毂和饰条的反射；不要写车型、品牌、车身颜色、车门开合、车牌或车的位置。没有车就省略这一段。\n"
    "不要描述构图、机位、视角、焦段、画面布局或物体在画面中的位置；除「车身质感」外不要提到参考图里的车辆或人物。"
    "最后单独一句写「不要出现：」列出这张图里没有、但同类场景常被加上的元素（例如画面内的太阳、眩光、古堡遗迹、云海）。"
    "只输出描述本身，不要标题。"
)
# 1.1.0: products that are not vehicles (chairs, sofas, lamps…) get a describer that never
# talks about cars, so no 车 wording can leak into their image prompt.
ENV_DESCRIBE_PROMPT_PRODUCT = (
    "你是产品广告摄影的灯光与美术指导。请只阅读这张环境参考图，写一段 150–220 字的中文环境描述，"
    "供另一张图严格按此重建环境（之后会把一件产品放进这个环境里）。要求精准、克制，只写图中确实可见的内容，不推测、不美化、不补充。依次写清：\n"
    "1. 场景与背景元素：逐项列出可见的山体、建筑、植被、路牌、护栏等，并写明数量级和远近；\n"
    "2. 地面：材质、颜色、干湿、积雪或积水情况；\n"
    "3. 天气与天空：晴/阴/雾/雪，云量，天空颜色；\n"
    "4. 光线：主光方向（画面左/右/前/后/顶）、太阳高度、太阳是否出现在画面内、光质软硬、阴影长短；\n"
    "5. 色温与主色调、反差、大气通透度与氛围。\n"
    "6. 若图中有一个明显的主体物品：另起一段以「表面质感：」开头，写 40–100 字，只写它表面的光泽、高光位置和形状、"
    "环境在表面上的倒影、暗部深浅和明暗过渡；不要写它是什么物品、品牌、颜色、造型或位置。没有就省略这一段。\n"
    "不要描述构图、机位、视角、焦段、画面布局或物体在画面中的位置；不要提到参考图里的车辆、交通工具或人物，"
    "也不要使用“车”“车身”“车轮”“车漆”等词——道路只写成“路面”或“地面”。"
    "最后单独一句写「不要出现：」列出这张图里没有、但同类场景常被加上的元素（例如画面内的太阳、眩光、古堡遗迹、云海）。"
    "只输出描述本身，不要标题。"
)
_ENV_LOCK = threading.Lock()


def _file_md5(path):
    import hashlib
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def describe_environment(env_paths, *, auth, auth_path, client_version="", model="", cache_dir="",
                         timeout=180, cancel_check=None, log=None, vehicle=True):
    """Read the environment reference(s) with a text-only call; return one Chinese description.

    The reference images are never sent to the image model: only this text is.
    Results are cached per file hash in ``cache_dir/env_descriptions.json``.
    """
    paths = [str(p) for p in env_paths or [] if p and Path(p).is_file()]
    if not paths:
        return "", auth
    cache_file = Path(cache_dir) / "env_descriptions.json" if cache_dir else None
    base = os.environ.get(BASE_URL_ENV, "").strip() or DEFAULT_BASE_URL
    url = base.rstrip("/") + "/responses"
    texts = []
    with _ENV_LOCK:
        cache = {}
        if cache_file and cache_file.is_file():
            try:
                cache = json.loads(cache_file.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                cache = {}
        for path in paths:
            key = _file_md5(path) + (":v2" if vehicle else ":p1")
            if cache.get(key):
                texts.append(cache[key])
                continue
            url_img, _mime, _n = data_url(path)
            payload = {
                "model": (model or "").strip() or DEFAULT_MODEL,
                "instructions": "Describe the environment in the image as asked. Do not use any tool.",
                "input": [{"type": "message", "role": "user", "content": [
                    {"type": "input_text", "text": ENV_DESCRIBE_PROMPT if vehicle else ENV_DESCRIBE_PROMPT_PRODUCT},
                    {"type": "input_image", "image_url": url_img}]}],
                "tools": [], "parallel_tool_calls": False, "store": False, "stream": True, "include": [],
            }
            refreshed = False
            while True:
                t0 = time.time()
                try:
                    text = post_responses(url, auth_headers(auth, client_version), payload, timeout=timeout,
                                          cancel_check=cancel_check, want="text")
                    break
                except DirectEditError as exc:
                    if exc.kind == "auth" and exc.status == 401 and not refreshed:
                        refreshed = True
                        auth = refresh_auth(auth, auth_path)
                        continue
                    raise
            text = re.sub(r"\s+", " ", text).strip()[:1300]
            cache[key] = text
            texts.append(text)
            if log is not None:
                log.setdefault("env_describe", []).append(
                    {"file": Path(path).name, "seconds": round(time.time() - t0, 1), "text": text})
        if cache_file:
            try:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                cache_file.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
            except OSError:
                pass
    return "\n".join(texts), auth


# --------------------------------------------------------------------------- main entry

def edit_image(*, prompt, edit_base_path, reference_paths, output_path, size, mask_path="", manifest=None,
               repair=False, model="", client_version="", timeout=900, log_dir="", tag="",
               cancel_check=None, correction="", simple_brief=None, terrain_text="", vehicle=False,
               exact_prompt=False, massing_text="", sketch_refs=None):
    """Edit ``edit_base_path`` (image 1) and save one PNG to ``output_path``.

    ``simple_brief`` (not None) = simple path: only the canvas is sent, with a
    short scene prompt, like a hand-made edit on the clay render.
    ``exact_prompt`` sends ``prompt`` verbatim (used by the 分层输出 clean plate).
    ``sketch_refs`` = [(path, label)] (草图模式, simple path only): the three-view images
    are attached after the clay canvas as product appearance references; environment
    references are still read as text only.
    """
    started = time.time()
    if not edit_base_path or not Path(edit_base_path).is_file():
        raise DirectEditError("编辑底图不存在", kind="input")
    auth_path = find_auth_file()
    if auth_path is None:
        raise DirectEditError("未找到 Codex 登录文件（~/.codex/auth.json 或 CODEX_HOME）", kind="auth", fatal=True)
    auth = load_auth(auth_path)
    if token_expiring(access_token(auth)):
        auth = refresh_auth(auth, auth_path)
    chain = model_chain(model, available_models(auth, auth_path, client_version))
    simple = simple_brief is not None and not repair
    if simple:
        images = [(str(Path(edit_base_path).resolve()), "camera", "白模底图")]
        labels = _labels_from_manifest(manifest)
        env = []
        dropped = []
        for ref in reference_paths or []:
            rp = str(Path(ref).resolve())
            role = labels.get(rp, ("", ""))[0]
            if role == "environment" and len(env) < SIMPLE_ENV_LIMIT and rp not in env:
                env.append(rp)
            else:
                dropped.append((rp, "simple"))
        # Environment refs are read as text only; the image model sees just the clay canvas
        # (plus, in 草图模式, the three-view images that define the product).
        env_paths = env
        for sp, slabel in list(sketch_refs or [])[:SKETCH_REF_LIMIT]:
            if sp and Path(sp).is_file() and len(images) < MAX_EDIT_IMAGES:
                images.append((str(Path(sp).resolve()), "product", f"产品三视图 · {slabel}"))
    else:
        images, dropped = select_images(edit_base_path, reference_paths, manifest, repair=repair)
    if not simple:
        env_paths = []
    sketch_labels = [label.split(" · ", 1)[-1] for _p, role, label in images[1:] if role == "product"] if simple else []
    log_root = Path(log_dir or Path(output_path).parent)
    mask, mask_note = prepare_mask(mask_path, _png_size(images[0][0]), log_root)
    env_text, env_note = "", ""
    if simple and env_paths:
        try:
            env_text, auth = describe_environment(env_paths, auth=auth, auth_path=auth_path,
                                                  client_version=client_version, model=chain[0],
                                                  cache_dir=str(log_root), cancel_check=cancel_check,
                                                  vehicle=bool(vehicle))
        except DirectEditError as exc:
            if exc.kind == "cancelled":
                raise
            env_note = "环境描述失败：" + str(exc)[:160]
    if exact_prompt:
        text = prompt  # 1.2.0 分层输出: the clean-plate edit sends its own complete instruction
    elif simple:
        text = build_simple_prompt(simple_brief, prompt, env_text=env_text, terrain_text=terrain_text,
                                   vehicle=bool(vehicle), massing_text=massing_text, sketch_views=sketch_labels)
    else:
        text = build_edit_prompt(prompt, images, mask=bool(mask), repair=repair, correction=correction)
    base = os.environ.get(BASE_URL_ENV, "").strip() or DEFAULT_BASE_URL
    url = base.rstrip("/") + "/responses"
    log = {"tag": tag, "url_host": re.sub(r"^(https?://[^/]+).*$", r"\1", url), "size": size,
           "images": [], "dropped": [Path(p).name + f"({r})" for p, r in dropped], "mask_note": mask_note,
           "attempts": [], "simple": simple, "vehicle": bool(vehicle), "prompt": text[:6000],
           "env_refs_as_text": [Path(p).name for p in env_paths] if simple else [], "env_note": env_note,
           "sketch_views": sketch_labels}
    model_used = chain[0]
    log["model_chain"] = chain[:6]
    variants = [dict(compat=False, model=m) for m in chain[:6]]
    variants.append(dict(compat=True, model=model_used))
    refreshed = False
    rate_waits = 0
    result = None
    last_exc = None
    index = 0
    try:
        while index < len(variants):
            opts = variants[index]
            payload, summary = build_payload(text, images, size=size, mask_path=mask, model=opts["model"], compat=opts["compat"],
                                             instructions=SKETCH_INSTRUCTIONS if sketch_labels and not exact_prompt else "")
            headers = auth_headers(auth, client_version)
            log["images"] = summary
            log["payload"] = redact_payload(payload)
            log["headers"] = redact_headers(headers)
            t0 = time.time()
            try:
                result = post_responses(url, headers, payload, timeout=timeout, cancel_check=cancel_check)
                log["attempts"].append({"ok": True, "compat": opts["compat"], "model": opts["model"],
                                        "seconds": round(time.time() - t0, 1)})
                break
            except DirectEditError as exc:
                last_exc = exc
                log["attempts"].append({"ok": False, "compat": opts["compat"], "model": opts["model"],
                                        "kind": exc.kind, "status": exc.status, "error": str(exc)[:300],
                                        "detail": getattr(exc, "detail", "")[:300],
                                        "seconds": round(time.time() - t0, 1)})
                if exc.kind == "cancelled":
                    raise
                if exc.kind == "auth" and exc.status == 401 and not refreshed:
                    refreshed = True
                    auth = refresh_auth(auth, auth_path)
                    continue
                if exc.kind == "rate_limit" and rate_waits < 2:
                    rate_waits += 1
                    time.sleep(min(30.0, 4.0 * rate_waits))
                    continue
                if exc.kind == "request" and index + 1 < len(variants):
                    index += 1  # 400: try a leaner, more compatible payload
                    continue
                if exc.kind == "auth":
                    exc.fatal = True
                raise
    finally:
        log["seconds"] = round(time.time() - started, 1)
        try:
            log_root.mkdir(parents=True, exist_ok=True)
            name = f"codex_direct_{tag}.json" if tag else "codex_direct.json"
            (log_root / name).write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError:
            pass
    b64, item, response_id = result
    try:
        data = base64.b64decode(b64, validate=False)
    except (binascii.Error, ValueError) as exc:
        raise DirectEditError("图像结果不是有效的 base64", kind="parse") from exc
    if sniff_mime(data) == "application/octet-stream":
        raise DirectEditError("图像结果无法识别", kind="parse")
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    from .composition import image_dimensions
    return {
        "engine": "codex_direct_edit",
        "status": "ok",
        "source": str(target),
        "attached_images": len(images),
        "image_roles": [role for _p, role, _l in images],
        "dropped_images": [r for _p, r in dropped],
        "mask": bool(mask),
        "mask_note": mask_note,
        "simple": simple,
        "sketch_views": sketch_labels,
        "size_requested": size,
        "size_returned": list(image_dimensions(target) or []),
        "model": log["attempts"][-1]["model"] if log["attempts"] else model_used,
        "compat_payload": bool(log["attempts"] and log["attempts"][-1].get("compat")),
        "response_id": response_id,
        "revised_prompt": str(item.get("revised_prompt") or "")[:500],
        "seconds": round(time.time() - started, 1),
    }


def short_reason(exc) -> str:
    text = str(exc)
    return text if len(text) <= 60 else text[:58] + "…"


def generate_with_fallback(kwargs, *, fallback, state, size, manifest=None, bundle_note="",
                           client_version="", on_status=None, cancel_check=None,
                           simple_base="", simple_brief=None, terrain_text="", vehicle=False, massing_text="", sketch_refs=None):
    """Try the direct edit; on any failure log it and run ``fallback(**kwargs)`` (codex exec).

    ``state`` is a per-render dict: once a fatal error happens (auth, endpoint),
    direct edit is skipped for the remaining variants/attempts of this render.
    """
    state.setdefault("direct_ok", 0)
    state.setdefault("fallbacks", 0)
    state.setdefault("disabled", "")
    state.setdefault("errors", [])
    edit_base = kwargs.get("edit_base_path") or (list(kwargs.get("reference_paths") or [""])[0])
    references = list(kwargs.get("reference_paths") or [])
    repair = bool(edit_base and references and Path(edit_base).resolve() != Path(references[0]).resolve())
    use_simple = simple_brief is not None and not repair and simple_base and Path(simple_base).is_file()
    if use_simple:
        edit_base = simple_base
    if not state["disabled"]:
        prompt = kwargs.get("prompt", "")
        if bundle_note:
            prompt = prompt.replace(bundle_note, "")
        out = Path(kwargs["output_path"])
        try:
            result = edit_image(
                prompt=prompt, edit_base_path=edit_base, reference_paths=references,
                output_path=str(out), size=size, mask_path=kwargs.get("edit_mask_path", ""),
                manifest=manifest, repair=repair, model=kwargs.get("model", ""),
                client_version=client_version, timeout=max(60, int(kwargs.get("timeout") or 900) or 900),
                log_dir=kwargs.get("cwd", ""), tag=out.stem, cancel_check=cancel_check,
                simple_brief=simple_brief if use_simple else None,
                terrain_text=terrain_text if use_simple else "",
                vehicle=bool(vehicle),
                massing_text=massing_text if use_simple else "",
                sketch_refs=sketch_refs if use_simple else None,
            )
            state["direct_ok"] += 1
            return result
        except DirectEditError as exc:
            if exc.kind == "cancelled":
                raise RuntimeError("任务已取消") from exc
            reason = short_reason(exc)
            state["errors"].append({"kind": exc.kind, "status": exc.status, "message": str(exc)[:300],
                                    "detail": getattr(exc, "detail", "")[:300]})
            if exc.fatal:
                state["disabled"] = reason
            state["last_reason"] = reason
            print(f"[Wondful] Codex 直连图像编辑失败，改用 codex exec：{reason}")
            if on_status:
                try:
                    on_status(f"直连编辑失败（{reason}），改用 Codex 代理生成")
                except Exception:
                    pass
        except Exception as exc:  # never let an unexpected bug block rendering
            reason = short_reason(f"{type(exc).__name__}: {exc}")
            state["errors"].append({"kind": "bug", "message": reason})
            state["disabled"] = reason
            state["last_reason"] = reason
            print(f"[Wondful] Codex 直连图像编辑异常，改用 codex exec：{reason}")
    state["fallbacks"] += 1
    result = fallback(**kwargs)
    if isinstance(result, dict):
        result = dict(result)
        result["direct_fallback_reason"] = state.get("last_reason") or state.get("disabled") or ""
    return result


def engine_note(state) -> str:
    """One short panel line describing which path produced the images."""
    if not state:
        return ""
    ok, fb = int(state.get("direct_ok", 0)), int(state.get("fallbacks", 0))
    reason = state.get("disabled") or state.get("last_reason") or ""
    if ok and not fb:
        return "生图：Codex 图像编辑（直连）"
    if fb and not ok:
        return f"直连编辑不可用（{reason}），已自动改用 Codex 代理生成" if reason else "已使用 Codex 代理生成"
    if ok and fb:
        return f"直连 {ok} 次，{fb} 次改用 Codex 代理生成（{reason}）"
    return ""
