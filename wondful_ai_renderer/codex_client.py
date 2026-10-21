from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .cli_runtime import runtime_env, find_cli
from .image_artifacts import ArtifactTracker
from .prompt_engine import REFERENCE_LIGHTING_POLICY


class CodexError(RuntimeError):
    pass


class CodexNotInstalledError(CodexError):
    pass


class CodexNotLoggedInError(CodexError):
    pass


class CodexModelRejectedError(CodexError):
    """1.2.2: the local Codex CLI / backend refused the chosen 推理 model (old CLI, new model)."""


_MODEL_REJECT_RE = re.compile(
    r"model is not supported|not supported when using codex|model_not_found|unsupported model|unknown model"
    r"|invalid model|model .{0,80}(?:does not exist|not found|is not available|requires a newer)"
    r"|requires a newer version of codex|please (?:update|upgrade) codex",
    re.I,
)
# 1.2.2 run-time fallback chain for the 推理 model.  Set on Blender's main thread
# (operators._analysis_provider) from the merged model list; read by worker threads.
_MODEL_STATE = {"chain": [], "notice": ""}


def is_model_rejection(detail: str) -> bool:
    return bool(_MODEL_REJECT_RE.search(detail or ""))


def set_model_fallbacks(chain) -> None:
    _MODEL_STATE["chain"] = [str(m).strip() for m in (chain or []) if str(m or "").strip()]


def pop_model_notice() -> str:
    notice, _MODEL_STATE["notice"] = _MODEL_STATE["notice"], ""
    return notice


def model_fallback_candidates(model: str, chain=None) -> list[str]:
    """Models to try after ``model`` was rejected: the rest of the list (newest first,
    starting after the rejected one), then "" = the CLI's own default. At most 4."""
    model = (model or "").strip()
    chain = list(_MODEL_STATE["chain"] if chain is None else chain)
    if model in chain:
        chain = chain[chain.index(model) + 1:]
    out = []
    for slug in chain:
        if slug and slug != model and slug not in out and not slug.startswith("chatgpt-web/"):
            out.append(slug)
    return out[:3] + [""]


class CodexImageGenerationUnavailable(CodexError):
    pass


@dataclass(frozen=True)
class CodexAccountStatus:
    installed: bool
    logged_in: bool
    cli_path: str = ""
    version: str = ""
    message: str = ""


def _runtime_env(cli_path="") -> dict[str, str]:
    return runtime_env(cli_path)


def _newest(paths):
    """Newest existing match first (extension folders carry version numbers)."""
    hits = [p for p in paths if p.is_file()]
    try:
        return sorted(hits, key=lambda p: p.stat().st_mtime, reverse=True)
    except OSError:
        return hits


def _codex_candidates() -> list[Path]:
    """Places Codex lives when Blender's PATH does not include it (3.1.9).

    Covers npm / nvm / volta / scoop installs, the Codex desktop app, and the
    codex binary bundled inside the VS Code / Cursor / Windsurf ChatGPT extension.
    """
    home = Path.home()
    system = platform.system()
    candidates: list[Path] = []
    ext_roots = [home / ".vscode/extensions", home / ".vscode-insiders/extensions",
                 home / ".cursor/extensions", home / ".windsurf/extensions"]
    if system == "Windows":
        env = os.environ
        appdata = Path(env.get("APPDATA", home / "AppData/Roaming"))
        local = Path(env.get("LOCALAPPDATA", home / "AppData/Local"))
        program_files = [Path(env.get(k)) for k in ("ProgramFiles", "ProgramFiles(x86)", "ProgramW6432") if env.get(k)]
        names = ("codex.cmd", "codex.exe", "codex.ps1")
        dirs = [appdata / "npm", home / "scoop/shims", home / ".bun/bin", home / ".cargo/bin",
                local / "Volta/bin", local / "pnpm", local / "Microsoft/WindowsApps",
                local / "Programs/codex", local / "Programs/Codex", local / "OpenAI/Codex/bin"]
        if env.get("NVM_SYMLINK"):
            dirs.insert(0, Path(env["NVM_SYMLINK"]))
        for pf in program_files:
            dirs += [pf / "nodejs", pf / "Codex", pf / "OpenAI/Codex"]
        for d in dirs:
            candidates += [d / n for n in names if n != "codex.ps1"]
        nvm_home = Path(env.get("NVM_HOME", appdata / "nvm"))
        try:
            candidates += _newest(nvm_home.glob("v*/codex.cmd"))
        except OSError:
            pass
        for root in ext_roots:
            try:
                candidates += _newest(root.glob("openai.chatgpt-*/bin/windows-*/codex.exe"))
            except OSError:
                pass
        for base in (local / "Programs", local):
            try:
                candidates += _newest(base.glob("*odex*/**/codex.exe"))
            except OSError:
                pass
    else:
        if system == "Darwin":
            for apps in (Path("/Applications"), home / "Applications"):
                candidates.extend([apps / "ChatGPT.app/Contents/Resources/codex",
                                   apps / "Codex.app/Contents/Resources/codex",
                                   apps / "Codex.app/Contents/MacOS/codex"])
        for root in ext_roots:
            try:
                candidates += _newest(root.glob("openai.chatgpt-*/bin/*/codex"))
            except OSError:
                pass
    return candidates


def discover_codex_cli(explicit_path: str = "") -> str | None:
    return find_cli("codex", explicit_path, ("CODEX_CLI_PATH",), _codex_candidates())


def _run(
    cli_path: str,
    args: list[str],
    *,
    input_text: str | None = None,
    cwd: str | None = None,
    timeout: int | None = 120,
) -> subprocess.CompletedProcess:
    effective_timeout = None if timeout is None or int(timeout) <= 0 else int(timeout)
    try:
        return subprocess.run(
            [cli_path, *args],
            input=input_text,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
            env=_runtime_env(cli_path),
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=effective_timeout,
        )
    except FileNotFoundError as exc:
        raise CodexNotInstalledError("未找到 Codex CLI。请先安装 Codex，或在插件偏好设置中指定 Codex 可执行文件。") from exc
    except subprocess.TimeoutExpired as exc:
        raise CodexError(
            f"Codex 操作超过插件允许的总时长（{effective_timeout}s），已被插件终止。"
            "这不代表 Codex 登录失效；可在插件偏好设置里提高对应超时，或设为 0 关闭插件侧硬超时。"
        ) from exc
    except Exception as exc:
        raise CodexError(f"无法启动 Codex: {exc}") from exc


def _combined_output(proc: subprocess.CompletedProcess) -> str:
    return "\n".join(s.strip() for s in (proc.stdout or "", proc.stderr or "") if s and s.strip()).strip()


def get_account_status(explicit_path: str = "") -> CodexAccountStatus:
    cli = discover_codex_cli(explicit_path)
    if not cli:
        return CodexAccountStatus(
            installed=False,
            logged_in=False,
            message="未检测到 Codex CLI",
        )

    version_proc = _run(cli, ["--version"], timeout=15)
    version = (version_proc.stdout or version_proc.stderr or "").strip().splitlines()
    version_text = version[0] if version else ""

    status_proc = _run(cli, ["login", "status"], timeout=20)
    text = _combined_output(status_proc)
    lower = text.lower()
    logged_in = status_proc.returncode == 0 and (
        "logged in" in lower
        or "chatgpt" in lower
        or "authenticated" in lower
        or "已登录" in text
    )
    return CodexAccountStatus(
        installed=True,
        logged_in=logged_in,
        cli_path=cli,
        version=version_text,
        message=text or ("已登录" if logged_in else "未登录 ChatGPT"),
    )


def login_chatgpt(explicit_path: str = "", timeout: int = 600) -> CodexAccountStatus:
    cli = discover_codex_cli(explicit_path)
    if not cli:
        raise CodexNotInstalledError("未找到 Codex CLI。请先安装 Codex，或在插件偏好设置中指定 Codex 可执行文件。")
    proc = _run(cli, ["login"], timeout=timeout)
    if proc.returncode != 0:
        detail = _combined_output(proc)
        raise CodexError(f"Codex ChatGPT 登录失败。{detail}".strip())
    status = get_account_status(cli)
    if not status.logged_in:
        raise CodexNotLoggedInError(f"Codex 登录命令已结束，但未检测到已登录状态。{status.message}".strip())
    return status


def logout(explicit_path: str = "") -> CodexAccountStatus:
    cli = discover_codex_cli(explicit_path)
    if not cli:
        raise CodexNotInstalledError("未找到 Codex CLI。")
    proc = _run(cli, ["logout"], timeout=30)
    if proc.returncode != 0:
        raise CodexError(f"Codex 退出登录失败。{_combined_output(proc)}".strip())
    return get_account_status(cli)


def _assert_logged_in(cli: str) -> None:
    status = get_account_status(cli)
    if not status.logged_in:
        raise CodexNotLoggedInError("Codex 尚未使用 ChatGPT 登录。请先点击“使用 ChatGPT 登录”。")


def _exec_turn(*, model: str = "", **kwargs) -> str:
    """Run one codex exec turn; if the CLI rejects ``model`` fall back down the chain.

    The fallback is recorded in ``_MODEL_STATE["notice"]`` so the panel can tell the
    user to update Codex CLI (operators pop it into props.model_notice)."""
    try:
        return _exec_turn_once(model=model, **kwargs)
    except CodexModelRejectedError as first:
        if not (model or "").strip():
            raise
        last = first
        for candidate in model_fallback_candidates(model):
            try:
                text = _exec_turn_once(model=candidate, **kwargs)
            except CodexModelRejectedError as exc:
                last = exc
                continue
            used = candidate or "CLI 默认模型"
            _MODEL_STATE["notice"] = (f"本机 Codex CLI 不支持推理模型 {model.strip()}，已自动改用 {used}；"
                                      "请更新 Codex CLI（npm i -g @openai/codex）后即可使用新模型")
            print(f"[Wondful] {_MODEL_STATE['notice']}")
            return text
        raise CodexModelRejectedError(
            f"本机 Codex CLI 不支持所选推理模型 {model.strip()}，自动降级也失败。请更新 Codex CLI（npm i -g @openai/codex）。{last}")


def _exec_turn_once(
    *,
    explicit_path: str,
    prompt: str,
    image_paths: Iterable[str] = (),
    model: str = "",
    cwd: str,
    sandbox: str,
    timeout: int,
    enable_image_generation: bool = False,
    output_last_message: str | None = None,
) -> str:
    cli = discover_codex_cli(explicit_path)
    if not cli:
        raise CodexNotInstalledError("未找到 Codex CLI。请先安装 Codex，或在插件偏好设置中指定 Codex 可执行文件。")
    _assert_logged_in(cli)

    workdir = Path(cwd)
    workdir.mkdir(parents=True, exist_ok=True)
    message_path = Path(output_last_message) if output_last_message else workdir / "codex_last_message.txt"
    message_path.parent.mkdir(parents=True, exist_ok=True)
    message_path.unlink(missing_ok=True)

    args = [
        "exec",
        "--skip-git-repo-check",
        "--ephemeral",
        "--ignore-user-config",
        "--sandbox",
        sandbox,
        "--output-last-message",
        str(message_path),
    ]
    if model.strip():
        args += ["--model", model.strip()]
    if enable_image_generation:
        args += ["--enable", "image_generation"]
    for path in image_paths:
        if path and Path(path).exists():
            args += ["--image", str(Path(path).resolve())]
    # '-' explicitly tells codex exec to read the prompt from stdin. This avoids
    # platform-specific quoting issues with long multilingual prompts.
    args += ["-"]

    proc = _run(cli, args, input_text=prompt, cwd=str(workdir), timeout=timeout)
    if proc.returncode != 0:
        detail = _combined_output(proc)
        if "usage_limit_reached" in detail or "workspace is out of credits" in detail.lower():
            raise CodexError("Codex / ImageGen 当前工作区额度已用完（429）。请等待额度恢复或联系工作区管理员补充额度，再重新渲染。重新登录不会恢复额度。")
        if "not logged" in detail.lower() or "unauthorized" in detail.lower() or "401" in detail:
            raise CodexNotLoggedInError(f"Codex 登录已失效，请重新登录。{detail}".strip())
        if model.strip() and is_model_rejection(detail):
            raise CodexModelRejectedError(f"Codex 拒绝了模型 {model.strip()}。{detail[-300:]}".strip())
        raise CodexError(f"Codex 执行失败（exit={proc.returncode}）。{detail}".strip())

    text = ""
    try:
        if message_path.exists():
            text = message_path.read_text(encoding="utf-8", errors="replace").strip()
    except Exception:
        text = ""
    if not text:
        text = (proc.stdout or "").strip()
    return text


def polish_prompt(
    *,
    system_prompt: str,
    user_text: str,
    image_paths: Iterable[str],
    cwd: str,
    explicit_path: str = "",
    model: str = "",
    timeout: int = 900,
) -> str:
    prompt = (
        system_prompt.strip()
        + "\n\n--- 当前任务 ---\n"
        + user_text.strip()
        + "\n\n只输出最终可直接用于图像生成的提示词正文。不要解释分析过程，不要输出 Markdown 标题或代码块。请尽量组织成连续自然的一段正文，不要用空行分段，不要输出编号列表。"
    )
    text = _exec_turn(
        explicit_path=explicit_path,
        prompt=prompt,
        image_paths=image_paths,
        model=model,
        cwd=cwd,
        sandbox="read-only",
        timeout=timeout,
        enable_image_generation=False,
        output_last_message=str(Path(cwd) / "codex_polish.txt"),
    )
    if not text:
        raise CodexError("Codex 未返回润色后的提示词。")
    return text.strip()



def _extract_json_dict(text: str) -> dict:
    raw = (text or "").strip()
    if not raw:
        raise CodexError("Codex 构图校验没有返回内容。")
    # Prefer a fenced or plain JSON object. Keep parsing deliberately small and strict.
    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        raise CodexError(f"Codex 构图校验未返回 JSON：{raw[:300]}")
    try:
        data = json.loads(match.group(0))
    except Exception as exc:
        raise CodexError(f"无法解析 Codex 构图校验 JSON：{raw[:300]}") from exc
    if not isinstance(data, dict):
        raise CodexError("Codex 构图校验结果不是 JSON 对象。")
    return data


def audit_alignment(
    *,
    camera_reference: str,
    generated_image: str,
    cwd: str,
    explicit_path: str = "",
    model: str = "",
    timeout: int = 600,
    tag: str = "",
    structure_control: dict | None = None,
    structure_guide_path: str = "",
) -> dict:
    """Ask Codex Vision to judge structural alignment against Blender projection anchors."""
    structure_control = structure_control or {}
    structure_enabled = bool(structure_control.get("enabled"))
    identity_items = [str(x) for x in (structure_control.get("identity_checklist") or []) if str(x).strip()]
    identity_check = ""
    if identity_items:
        identity_check = (
            "\n另外逐项检查候选图是否清晰保留这些产品细节（不计入构图分数）："
            + "、".join(identity_items)
            + "。把缺失或明显变形的项写进 JSON 的 missing_details 数组，没有则为空数组。\n"
        )
    bbox = structure_control.get("bbox_normalized", [])
    center = structure_control.get("center_normalized", [])
    wheels = structure_control.get("wheel_points_normalized", [])
    scene_bbox = structure_control.get("scene_bbox_normalized", [])
    numeric_targets = ""
    if structure_enabled:
        numeric_targets = f"""
Blender 还提供了精确 Camera 投影锚点（坐标均为相对于最终画布的 0~1 归一化坐标，原点在左上）：
- scene_canvas_bbox [left, top, right, bottom] = {scene_bbox}
- primary_target_bbox [left, top, right, bottom] = {bbox}
- primary_target_center [x, y] = {center}
- target_wheel_anchors = {wheels}
structure_guide 只来自用户指定“产品集合”内、当前 Scene Camera 可见的 Mesh 真实投影；这些产品数值和结构线优先级高于视觉猜测。
"""

    prompt = f"""
你是 Wondful AI 渲染器的构图验收器。你会收到：
Reference 1 = Blender Camera Reference（唯一正确的构图/机位/空间基准）。
Reference 2 = AI 渲染候选图。
Reference 3（若存在）= Blender 产品轮廓 / 投影参考。
{numeric_targets}{identity_check}
只评估“几何与摄影机对齐”，不要因为候选图材质、灯光、画质更好就给高分。重点比较：
- 产品二维包围框中心位置、左右上下边界与占画面比例
- 前轮和后轮轮心在画面中的位置
- 产品正面/背面朝向（车辆即车头/车尾）、可见侧面比例、俯仰/偏航/滚转观感
- 相机高度、视角、焦段/透视压缩感
- 地平线、主要消失方向、道路/地面透视
- 前后遮挡与主体数量

如果存在 target_bbox/target_center/target_wheel_anchors，纠偏必须尽量量化，例如“产品中心向右移动约 2.5% 画宽、整体缩小约 4%、前侧接地点（车辆为前轮轮心）向下约 1.5% 画高”。不要只写“更接近参考图”。

返回且只返回一个 JSON 对象，不要 Markdown，不要解释：
{{
  "overall_score": 0到100的整数,
  "vehicle_position": 0到100的整数,
  "vehicle_scale": 0到100的整数,
  "viewpoint": 0到100的整数,
  "wheel_alignment": 0到100的整数,
  "horizon_perspective": 0到100的整数,
  "observed_bbox": [候选图主体的left, top, right, bottom] 或 null,
  "bbox_confidence": 0到1之间的数,
  "passed": true或false,
  "correction": "给下一次 image edit 用的一句中文纠偏指令；必须具体说明移动方向/百分比、缩放百分比、视角或轮心如何恢复；若已经高度对齐则写保持当前构图"
}}
observed_bbox 必须独立观察 Reference 2 中主体可见边界，坐标以其整幅画布归一化、原点左上，不包含影子，不要照抄 Blender 目标数值。不确定时返回 null，bbox_confidence 设为 0。该框只是视觉估计，不能声称是逐像素测量。轮部件锚点来自命名 Mesh 的包围盒中心估计，仅辅助观察，不能当作精确轮轴。
评分要严格：明显的产品位置、尺度或视角偏差不应超过 80 分。
""".strip()
    suffix = f"_{tag}" if tag else ""
    images = [camera_reference, generated_image]
    if structure_guide_path and Path(structure_guide_path).exists():
        images.append(structure_guide_path)
    text = _exec_turn(
        explicit_path=explicit_path,
        prompt=prompt,
        image_paths=images,
        model=model,
        cwd=cwd,
        sandbox="read-only",
        timeout=timeout,
        enable_image_generation=False,
        output_last_message=str(Path(cwd) / f"codex_alignment{suffix}.txt"),
    )
    data = _extract_json_dict(text)
    try:
        score = int(round(float(data.get("overall_score", 0))))
    except Exception:
        score = 0
    data["overall_score"] = max(0, min(100, score))
    data["correction"] = str(data.get("correction", "")).strip()
    missing = data.get("missing_details") or []
    if not isinstance(missing, list):
        missing = [str(missing)]
    missing = [str(x).strip() for x in missing if str(x).strip()]
    data["missing_details"] = missing
    if missing:
        fix = "补回并清晰呈现：" + "、".join(missing) + "。"
        data["correction"] = (str(data.get("correction", "")).strip() + " " + fix).strip()
    return data

def _extract_candidate_paths(text: str) -> list[Path]:
    if not text:
        return []
    patterns = [
        r"`([^`\n]+\.(?:png|jpg|jpeg|webp))`",
        r"(?m)(/[^\s\"'`]+\.(?:png|jpg|jpeg|webp))",
    ]
    found: list[Path] = []
    for pattern in patterns:
        for match in re.findall(pattern, text, flags=re.IGNORECASE):
            try:
                p = Path(match).expanduser()
                if p.exists() and p.is_file():
                    found.append(p)
            except Exception:
                continue
    return found


def generate_image(
    *,
    prompt: str,
    reference_paths: Iterable[str],
    output_path: str,
    cwd: str,
    explicit_path: str = "",
    model: str = "",
    timeout: int = 1800,
    edit_base_path: str = "",
    edit_mask_path: str = "",
    structure_control: dict | None = None,
) -> dict:
    from . import static_scene as _SS
    prompt = _SS.apply(prompt)   # 1.2.6 只做场景相关物体 (no pedestrians / other vehicles / animals)
    from .reference_bundle import check_image_capacity, reference_limit_error, ReferenceLimitError, unique_paths
    session_dir = Path(cwd).resolve()
    target = Path(output_path).resolve()
    refs = []
    for index, source in enumerate(reference_paths, start=1):
        if not source or not Path(source).expanduser().is_file():
            raise CodexError(f"Reference {index} 已不存在，请重新选择参考图。")
        refs.append(str(Path(source).expanduser().resolve()))
    for label, source in (("Edit Base", edit_base_path), ("Edit Mask", edit_mask_path)):
        if source and not Path(source).expanduser().is_file():
            raise CodexError(f"{label} 已不存在，请重新捕获结构图。")
    edit_base = str(Path(edit_base_path).resolve()) if edit_base_path and Path(edit_base_path).exists() else ""
    edit_mask = str(Path(edit_mask_path).resolve()) if edit_mask_path and Path(edit_mask_path).exists() else ""
    # Keep the documented reference numbering stable: camera_reference / structure /
    # product / person / style remain first.  Edit base + mask are appended as explicit
    # editing assets so a correction pass never shifts Reference 1 away from Blender.
    attached = list(refs)
    for candidate in ([edit_base] if edit_base else []) + ([edit_mask] if edit_mask else []):
        if candidate and candidate not in attached:
            attached.append(candidate)
    attached = unique_paths(attached)
    check_image_capacity(attached)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        try:
            target.unlink()
        except OSError:
            pass

    started_at = time.time()
    structure_note = ""
    packet_roles = []
    if structure_control and structure_control.get("enabled"):
        bbox = structure_control.get("bbox_normalized", [])
        center = structure_control.get("center_normalized", [])
        wheels = structure_control.get("wheel_points_normalized", [])
        mesh_count = structure_control.get("product_collection_mesh_count", structure_control.get("camera_visible_mesh_count", 0))
        collection_name = structure_control.get("product_collection_name", "")
        packet_roles = list(structure_control.get("generation_reference_roles", []))
        packet_version = structure_control.get("packet_version", "2.x")
        native_passes = bool(structure_control.get("native_passes", False))
        role_lines = [f"- Structure source role: {role}" for role in packet_roles]
        structure_note = f"""
Blender Structure Lock {packet_version}：
- Camera Base 是唯一整张画面的 Camera / Composition / Perspective 权威。
- 产品集合 = {collection_name}；参与产品结构的 Mesh 数量 = {mesh_count}。
- product_bbox_normalized = {bbox}；product_center_normalized = {center}；wheel anchors = {wheels}。
- native data passes = {native_passes}。
{chr(10).join(role_lines) if role_lines else '- 当前没有额外 Structure Packet 图片。'}
Structure Packet 全部是控制数据，不是最终外观：Mask 锁 footprint，Depth 锁空间层级，Normal 锁表面朝向，Silhouette 锁轮廓，Part ID 只锁 Mesh 部件区域。任何伪彩色都不得复制到最终图。
"""

    if edit_base and edit_mask:
        edit_note = f"""
这是局部构图修复轮次，不是重新从零生成：
- 编辑底图：{edit_base}
- Blender 产品编辑 Mask：{edit_mask}
必须优先使用 image edit / edit existing image 工作流，以编辑底图保留已经满意的材质、灯光和背景。
如果当前 image generation 工具 schema 支持 input_image_mask / mask，必须把上述 product_edit_mask 作为编辑 Mask；Mask 的透明区域是允许修复的产品区域，外部区域应尽量保持不变。
如果工具没有暴露显式 mask 参数，也必须把该 Mask 作为视觉区域约束，只修复产品位置、尺度、视角、轮廓和接地点，不要整张重新构图。
"""
    elif edit_base:
        edit_note = f"""
这是白模高保真重渲染：
- 基础画布：{edit_base}
必须优先使用 image edit / edit existing image / high-fidelity image input，把 Blender Camera Reference 作为 base image，在**同一画布、同一相机、同一主体投影位置**上替换材质、灯光、环境细节和摄影质感。
没有提供局部 Mask，表示本轮允许整幅画面的 Appearance 更新，但 Geometry / Position / Scale / Rotation / Perspective 不允许漂移。
"""
    else:
        edit_note = """
如果没有可用 edit base，才退回普通多参考图生成；仍必须把 camera_reference 作为最高优先级结构基准。
"""

    task = f"""
你是Wondful AI 渲染器的最终图像生成执行器。

必须使用当前 Codex 会话提供的内置 image generation / image_gen 能力生成 **恰好 1 张**最终图片；不要用 Python、SVG、HTML、Blender 渲染或其他方式伪造最终图片。

核心原则：Blender 负责 Structure，AI 只负责 Appearance。Camera / Composition / Geometry / Position / Scale / Rotation / Spatial Relationship 都以 Blender 为准。

参考图顺序：
1. Camera Base：Blender 当前 Scene Camera 的原位编辑底图，是整张画面的最高结构权威。
2. 随后的若干附件若属于 Blender Structure Packet，则只承担各自的 Mask / Depth / Normal / Silhouette / Part-ID 结构职责。
3. Structure Packet 后的 product 只提供产品身份与造型识别；person 提供人物身份；环境／style 提供照明及视觉语言，均不能推翻 Blender 结构。

{REFERENCE_LIGHTING_POLICY}

{structure_note}
{edit_note}
你不是在重新设计一张更好看的构图，而是在 Blender 原位置上替换 Appearance。禁止 zoom、pan、crop、reframe、换机位、改变焦段观感或重新安排产品。

最终生成要求：
{prompt.strip()}

执行要求：
- 使用内置 image generation 工具真正生成/编辑图片。
- 最终只生成一张图。
- 严格保持 Blender Camera Output 的宽高方向和构图比例。
- 产品身份参考只决定“产品长什么样”；camera_reference + structure guide/mask 决定“产品在哪里、占多大、什么视角”。
- 生成完成后，将最终位图复制或保存到这个绝对路径：
  {target}
- 在结束前必须检查该路径确实存在且是 PNG/JPG/WEBP 位图。
- 生图工具的 referenced_image_paths 只能使用本轮附件清单中的路径，最多 5 个；编辑底图已在清单中时不得重复追加。不要同时传 num_last_images_to_include。
- 本轮实际附件路径清单：{json.dumps(attached, ensure_ascii=False)}
- 只有没有暴露 image generation/image_gen 工具时，回复 `WONDFUL_IMAGEGEN_UNAVAILABLE:` 并说明原因。
- 参数超限、配额、工具执行失败或没有位图时，回复 `WONDFUL_IMAGEGEN_ERROR:` 并保留实际原因；不能据此判断 OAuth 无效。不要假装已经生成成功。
- 如果成功，最后只回复 `WONDFUL_IMAGEGEN_OK` 和实际文件路径。
""".strip()

    tracker = ArtifactTracker(session_dir, attached)
    last_message = _exec_turn(
        explicit_path=explicit_path,
        prompt=task,
        image_paths=attached,
        model=model,
        cwd=str(session_dir),
        sandbox="workspace-write",
        timeout=timeout,
        enable_image_generation=True,
        output_last_message=str(session_dir / "codex_render.txt"),
    )

    candidates = [target, *_extract_candidate_paths(last_message)]
    newest = tracker.newest()
    if newest is not None:
        candidates.append(newest)
    for candidate in candidates:
        if tracker.accepts(candidate):
            try:
                if candidate.resolve() != target:
                    shutil.copy2(candidate, target)
                return {"engine": "codex_oauth_imagegen", "status": "ok",
                        "message": last_message, "source": str(candidate),
                        "attached_images": len(attached)}
            except OSError:
                continue

    if reference_limit_error(last_message):
        raise ReferenceLimitError("图像工具拒绝了参考图数量；这不是登录失效。" + last_message[-700:])
    if "WONDFUL_IMAGEGEN_ERROR" in last_message:
        raise CodexError("Codex 图像工具执行失败：" + last_message.split("WONDFUL_IMAGEGEN_ERROR:", 1)[-1].strip())
    if "WONDFUL_IMAGEGEN_UNAVAILABLE" in last_message:
        reason = last_message.split("WONDFUL_IMAGEGEN_UNAVAILABLE:", 1)[-1].strip()
        raise CodexImageGenerationUnavailable(
            "当前 Codex 会话报告未暴露 ImageGen 工具。"
            + (f" Codex: {reason}" if reason else "")
        )
    raise CodexError(
        "Codex 已结束，但没有得到可读取的最终图片文件。请查看本轮工具结果和文件保存信息。"
        + (f" Codex 返回：{last_message[:500]}" if last_message else "")
    )
