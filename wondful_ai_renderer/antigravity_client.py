from __future__ import annotations

import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import unquote, urlparse

from .cli_runtime import runtime_env, find_cli
from .cli_transport import run_headless
from .image_artifacts import ArtifactTracker, IMAGE_SUFFIXES
from .prompt_engine import REFERENCE_LIGHTING_POLICY


class AntigravityError(RuntimeError):
    pass


class AntigravityNotInstalledError(AntigravityError):
    pass


class AntigravityNotLoggedInError(AntigravityError):
    pass


class AntigravityImageGenerationUnavailable(AntigravityError):
    pass


@dataclass(frozen=True)
class AntigravityAccountStatus:
    installed: bool
    logged_in: bool
    cli_path: str = ""
    version: str = ""
    auth_type: str = ""
    message: str = ""
    state: str = "UNKNOWN"


def _runtime_env(cli_path="") -> dict[str, str]:
    return runtime_env(cli_path)


def discover_antigravity_cli(explicit_path: str = "") -> str | None:
    return find_cli("agy", explicit_path, ("ANTIGRAVITY_CLI_PATH", "AGY_CLI_PATH"))


def _run(
    cli_path: str,
    args: list[str],
    *,
    cwd: str | None = None,
    timeout: int | None = 120,
) -> subprocess.CompletedProcess:
    try:
        if any(flag in args for flag in ("--prompt", "--print", "-p")):
            return run_headless(cli_path, args, cwd=cwd, env=_runtime_env(cli_path),
                                timeout=None if timeout is None or int(timeout) <= 0 else int(timeout))
        return subprocess.run(
            [cli_path, *args],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            cwd=cwd,
            env=_runtime_env(cli_path),
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=None if timeout is None or int(timeout) <= 0 else int(timeout),
        )
    except FileNotFoundError as exc:
        raise AntigravityNotInstalledError(
            "未找到 Antigravity CLI（agy）。请先安装官方 Antigravity CLI，或在插件偏好设置中指定 agy 可执行文件。"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise AntigravityError(f"Antigravity 操作超过插件允许的总时长（{timeout}s）。") from exc
    except Exception as exc:
        raise AntigravityError(f"无法启动 Antigravity CLI: {exc}") from exc


def _combined_output(proc: subprocess.CompletedProcess) -> str:
    return "\n".join(s.strip() for s in (proc.stdout or "", proc.stderr or "") if s and s.strip()).strip()


def _json_objects(text: str):
    raw = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text or "").lstrip("\ufeff")
    decoder = json.JSONDecoder()
    offset = 0
    while offset < len(raw):
        start = raw.find("{", offset)
        if start < 0:
            break
        try:
            value, end = decoder.raw_decode(raw, start)
        except ValueError:
            offset = start + 1
            continue
        offset = end
        if isinstance(value, dict):
            yield value


def _parse_json_output(text: str) -> dict:
    result = {}
    for obj in _json_objects(text):
        if obj.get("event") == "result" and isinstance(obj.get("result"), dict):
            result = obj["result"]
        elif "status" in obj or "response" in obj or "error" in obj:
            result = obj
    return result


def _stream_agent_text(text: str) -> str:
    """Collect final assistant text when agy leaves result.response empty.

    AGY 1.1.x can finish a successful tool-using turn with an empty response
    field even though its stream contains the final ``agent_response`` delta.
    Only assistant-response events are accepted; tool output and echoed input
    must never become a prompt or an audit result.
    """
    chunks: list[str] = []
    for obj in _json_objects(text):
        if obj.get("event") != "step_update" or not isinstance(obj.get("step_update"), dict):
            continue
        step = obj["step_update"]
        if step.get("step_type") != "agent_response":
            continue
        value = step.get("text_delta")
        if value is None:
            value = step.get("text")
        if isinstance(value, str) and value:
            chunks.append(value)
    return "".join(chunks).strip()


def _looks_like_auth_error(detail: str) -> bool:
    # OAuth/keyring mentions, 403 policy denials and probe prompt echoes are not
    # evidence of an expired login. Require an explicit authentication failure.
    lower = (detail or "").lower()
    return bool(re.search(
        r"authentication required|not authenticated|unauthenticated|unauthorized|"
        r"not logged in|login required|sign[ -]?in required|please (?:log|sign)[ -]?in|"
        r"(?:credentials?|(?:access |oauth )?token)[^\n]{0,25}(?:expired|invalid|missing|not found)|"
        r"(?:expired|invalid|missing|no valid)[^\n]{0,25}(?:credentials?|token)|"
        r"(?:http(?: status)?[ :]*|status(?: code)?[ :=]*)401\b|"
        r"未登录|登录已失效|身份验证失败", lower,
    ))


def _capacity_error_message(detail: str) -> str:
    """Turn AGY's quota/capacity response into an actionable user message."""
    compact = " ".join(str(detail or "").split())
    if not re.search(
        r"(?:\b429\b|resource[_ -]?exhausted|too many requests|"
        r"exhausted (?:your )?capacity|quota)",
        compact,
        flags=re.IGNORECASE,
    ):
        return ""
    reset = re.search(
        r"(?:quota|capacity)[^.]{0,120}?(?:will\s+)?reset\s+after\s+"
        r"([0-9]+h(?:[0-9]+m)?(?:[0-9]+s)?)",
        compact,
        flags=re.IGNORECASE,
    )
    if reset:
        return (
            "Antigravity 生图模型容量已耗尽（429 RESOURCE_EXHAUSTED），"
            f"服务端预计 {reset.group(1)} 后恢复。请等待后重试，"
            "或暂时切换到 Codex ImageGen；重新登录不会恢复配额。"
        )
    return (
        "Antigravity 生图模型容量已耗尽（429 RESOURCE_EXHAUSTED）。"
        "请等待额度恢复，或暂时切换到 Codex ImageGen；重新登录不会恢复配额。"
    )


def _exec_text(
    *,
    explicit_path: str,
    prompt: str,
    image_paths: Iterable[str] = (),
    model: str = "",
    cwd: str,
    timeout: int = 240,
) -> str:
    cli = discover_antigravity_cli(explicit_path)
    if not cli:
        raise AntigravityNotInstalledError("未找到 Antigravity CLI（agy）。请先安装官方 Antigravity CLI。")

    workdir = Path(cwd).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    staged = workdir / ("antigravity_inputs_" + uuid.uuid4().hex[:8])
    staged.mkdir(parents=True, exist_ok=True)

    refs: list[str] = []
    for index, source in enumerate(image_paths, start=1):
        if not source:
            continue
        src = Path(source).expanduser()
        if not src.is_file():
            raise AntigravityError(f"参考图已不存在：{src}")
        suffix = src.suffix.lower() if src.suffix else ".png"
        dest = staged / f"ref_{index:02d}{suffix}"
        try:
            if src.resolve() != dest.resolve():
                shutil.copy2(src, dest)
            # AGY 1.1.27 treats a relative path as a filename search request and
            # may scan the whole home directory instead of opening the staged
            # bitmap.  Pass the staged absolute path so view_file can resolve it
            # deterministically inside the isolated render session.
            refs.append(str(dest.resolve()))
        except OSError as exc:
            raise AntigravityError(f"Antigravity 参考图暂存失败：{src.name}: {exc}") from exc

    context_prefix = ""
    if refs:
        reference_lines = "\n".join(f"Reference {i + 1}: {ref}" for i, ref in enumerate(refs))
        context_prefix = (
            "你当前位于 Wondful AI 的隔离 Render Session 工作区。以下媒体文件已经复制到当前工作区内。"
            "必须使用可用的文件读取/视觉能力真正打开并分析每一张图片，禁止仅根据文件名推断内容。"
            "这些文件仅供读取，不要修改、删除或创建任何文件。Reference 顺序具有语义优先级：\n"
            + reference_lines
            + "\n\n"
        )
    full_prompt = context_prefix + prompt.strip()

    # Official headless contract: one prompt, streamable JSON response envelope,
    # cached credentials. Stream events let us recover final assistant text on
    # AGY builds that leave result.response empty after a tool call.
    # Workspace file reads are auto-allowed by Antigravity; we intentionally do not pass
    # --dangerously-skip-permissions because Wondful only needs read access here.
    args = [
        "--output-format",
        "stream-json",
        "--print-timeout",
        ("24h" if int(timeout) <= 0 else f"{max(30, int(timeout) - 10)}s"),
    ]
    if model.strip():
        args += ["--model", model.strip()]
    args += ["--prompt", full_prompt]

    proc = _run(cli, args, cwd=str(workdir), timeout=(None if int(timeout) <= 0 else int(timeout)))
    data = _parse_json_output(proc.stdout or "")
    status = str(data.get("status") or "").strip().upper() if isinstance(data, dict) else ""
    error = str(data.get("error") or "").strip() if isinstance(data, dict) else ""

    if proc.returncode != 0 or (status and status != "SUCCESS") or error:
        detail = error or _combined_output(proc)
        if _looks_like_auth_error(detail):
            raise AntigravityNotLoggedInError(f"Antigravity Google 登录不可用或已失效。{detail}".strip())
        raise AntigravityError(f"Antigravity CLI 执行失败（exit={proc.returncode}）。{detail}".strip())

    response = data.get("response") if isinstance(data, dict) else None
    if isinstance(response, (dict, list)):
        text = json.dumps(response, ensure_ascii=False)
    else:
        text = str(response).strip() if response is not None else ""
    if not text:
        text = _stream_agent_text(proc.stdout or "")
    if not text and not data:
        # Text fallback helps with early/alternate builds while still preferring the official JSON envelope.
        text = (proc.stdout or "").strip()
    if not text:
        raise AntigravityError("Antigravity CLI 没有返回模型响应。")
    return text


def get_account_status(explicit_path: str = "", *, live_check: bool = False) -> AntigravityAccountStatus:
    cli = discover_antigravity_cli(explicit_path)
    if not cli:
        return AntigravityAccountStatus(installed=False, logged_in=False, message="未检测到 Antigravity CLI（agy）")

    version = ""
    try:
        version_proc = _run(cli, ["--version"], timeout=15)
        version_lines = (version_proc.stdout or version_proc.stderr or "").strip().splitlines()
        version = version_lines[0] if version_lines else ""
    except AntigravityError:
        pass  # A version banner failure must not suppress the real auth probe.

    if live_check:
        try:
            probe_dir = Path(tempfile.gettempdir()) / "wondful_antigravity_auth_probe"
            probe_dir.mkdir(parents=True, exist_ok=True)
            args = [
                "--output-format", "json",
                "--print-timeout", "60s",
                "--prompt", "这是 Wondful AI 渲染器的登录状态探测。请简短回复 OK，不要调用工具。",
            ]
            proc = _run(cli, args, cwd=str(probe_dir), timeout=70)
            data = _parse_json_output(proc.stdout or "")
            status = str(data.get("status") or "").strip().upper() if isinstance(data, dict) else ""
            detail = str(data.get("error") or "").strip() if isinstance(data, dict) else ""
            if proc.returncode == 0 and status == "SUCCESS" and not detail:
                # A SUCCESS JSON envelope means agy completed an authenticated model turn.
                # Do not require a magic response string: some agy builds may deny a
                # harmless tool action (e.g. ListDir) yet still report SUCCESS.
                return AntigravityAccountStatus(
                    installed=True, logged_in=True, cli_path=cli, version=version,
                    auth_type="google-keyring-oauth", message="已通过 Antigravity / Google 账号登录", state="LOGGED_IN",
                )
            combined = detail or _combined_output(proc)
            if _looks_like_auth_error(combined):
                raise AntigravityNotLoggedInError(f"Antigravity Google 登录不可用或已失效。{combined}".strip())
            raise AntigravityError(f"Antigravity 登录探测失败（exit={proc.returncode}, status={status or 'UNKNOWN'}）。{combined}".strip())
        except AntigravityNotLoggedInError as exc:
            return AntigravityAccountStatus(
                installed=True, logged_in=False, cli_path=cli, version=version,
                auth_type="google-keyring-oauth", message=str(exc), state="LOGGED_OUT",
            )
        except AntigravityError as exc:
            return AntigravityAccountStatus(
                installed=True, logged_in=False, cli_path=cli, version=version,
                auth_type="google-keyring-oauth", message="检测未完成（不代表未登录）：" + str(exc), state="ERROR",
            )

    return AntigravityAccountStatus(
        installed=True,
        logged_in=False,
        cli_path=cli,
        version=version,
        auth_type="google-keyring-oauth",
        message="已检测到 agy。本次仅检查安装，不发起登录；点‘连接 / 验证 Google’确认连接。",
    )


def open_google_login(explicit_path: str = "") -> str:
    """Open interactive `agy`; authentication remains owned by Google's CLI/keyring."""
    cli = discover_antigravity_cli(explicit_path)
    if not cli:
        raise AntigravityNotInstalledError(
            "未找到 Antigravity CLI（agy）。请先按官方方式安装 Antigravity CLI。"
        )

    from .config_manager import CONFIG_DIR
    # Login and verification must use the same persistent workspace, so an
    # interactive first-run/trust choice is not lost to a fresh temporary folder.
    work = CONFIG_DIR
    work.mkdir(parents=True, exist_ok=True)
    system = platform.system()
    if system == "Darwin":
        script = work / "antigravity_google_login.command"
        script.write_text(
            "#!/bin/zsh\n"
            "echo 'Wondful AI - Antigravity Google Login'\n"
            "echo 'Antigravity 会优先复用 Apple Keychain；若没有有效登录，会自动打开浏览器完成 Google 登录。'\n"
            "echo '完成登录和首次设置后，回 Blender 点击验证登录。'\n"
            f"cd {shlex.quote(str(work))} || exit 1\n"
            f"export PATH={shlex.quote(_runtime_env(cli).get('PATH', ''))}\n"
            f"exec {shlex.quote(cli)}\n",
            encoding="utf-8",
        )
        script.chmod(0o700)
        opened = subprocess.run(["open", "-a", "Terminal", str(script)], env=_runtime_env(cli),
                                capture_output=True, text=True, timeout=15)
        if opened.returncode:
            raise AntigravityError("系统终端未能启动：" + (opened.stderr or "请手动打开 Terminal 运行 agy。"))
        return "已打开 Antigravity CLI。若浏览器要求登录请完成 Google 登录，然后回 Blender 点击连接验证。"

    if system == "Windows":
        flags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
        subprocess.Popen([cli], cwd=str(work), env=_runtime_env(cli), creationflags=flags)
        return "已打开 Antigravity CLI。完成 Google 登录后回 Blender 点击连接验证。"

    terminals = [
        ["x-terminal-emulator", "-e", cli],
        ["gnome-terminal", "--", cli],
        ["konsole", "-e", cli],
    ]
    for command in terminals:
        if shutil.which(command[0]):
            subprocess.Popen(command, cwd=str(work), env=_runtime_env(cli))
            return "已打开 Antigravity CLI。完成 Google 登录后回 Blender 点击连接验证。"
    raise AntigravityError("无法自动打开终端。请手动在 Terminal 运行 `agy`，完成登录后回 Blender 点击连接验证。")




def _collect_image_candidates(value, workdir: Path) -> list[Path]:
    """Extract image paths from Antigravity stream-json tool output defensively."""
    results: list[Path] = []
    seen: set[str] = set()

    def add_candidate(raw: str) -> None:
        raw = (raw or "").strip().strip('"\'` ,;')
        if not raw:
            return
        if raw.startswith(("{", "[")):
            try:
                walk(json.loads(raw))
                return
            except ValueError:
                pass
        # AGY may return a POSIX path, a file:// URL, or a sentence containing
        # either one. Decode URL-escaped spaces and tolerate JSON's escaped
        # forward slashes before resolving it against the isolated workspace.
        raw = raw.replace("\\/", "/")
        matches = []
        if re.search(r"\.(?:png|jpe?g|webp|bmp|tiff?)$", raw, flags=re.I):
            matches.append(raw)
        # Paths in AGY prose start with '/', '~/' or a Windows drive. Limiting
        # the expression to those prefixes avoids treating the whole sentence
        # (e.g. 'Generated image is saved at ...') as a filename.
        matches.extend(re.findall(
            r"(?:file://[^\n\r\t\"'<>]+|(?:[A-Za-z]:[\\/]|/|~[\\/])[^\n\r\t\"'<>]*?\.(?:png|jpe?g|webp|bmp|tiff?))",
            raw,
            flags=re.I,
        ))
        # A bare relative filename is also common in tool output.
        if not matches and re.fullmatch(r"[^/\\\s]+\.(?:png|jpe?g|webp|bmp|tiff?)", raw, flags=re.I):
            matches = [raw]
        for match in matches:
            text = match.strip().strip('"\'` ,;.\n\r')
            if not text:
                continue
            if text.lower().startswith("file:"):
                parsed = urlparse(text)
                if parsed.scheme == "file":
                    text = unquote(parsed.path or parsed.netloc)
            candidate = Path(text).expanduser()
            if not candidate.is_absolute():
                candidate = workdir / candidate
            try:
                key = str(candidate.resolve())
            except OSError:
                key = str(candidate)
            if key not in seen:
                seen.add(key)
                results.append(candidate)

    def walk(obj) -> None:
        if isinstance(obj, dict):
            for v in obj.values():
                walk(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                walk(v)
        elif isinstance(obj, str):
            add_candidate(obj)

    walk(value)
    return results


def _brain_image_candidates(
    conversation_id: str,
    brain_root: Path | None = None,
    preferred_name: str = "",
) -> list[Path]:
    """Return bitmap artifacts owned by one completed AGY conversation.

    AGY 1.1.x stores generate_image results in its conversation brain directory,
    while stream-json can expose only a short tool summary. Restrict the lookup
    to the opaque conversation ID and its descendants; never inspect credentials
    or another conversation. Some AGY builds put the bitmap in
    ``.tempmediaStorage`` below the conversation directory rather than beside
    the transcript, so the lookup must be recursive.
    """
    conversation_id = str(conversation_id or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{7,127}", conversation_id):
        return []
    root = Path(brain_root or (Path.home() / ".gemini" / "antigravity-cli" / "brain"))
    try:
        root = root.resolve()
        folder = (root / conversation_id).resolve()
        if not folder.is_relative_to(root) or not folder.is_dir():
            return []
        candidates = [
            path
            for path in folder.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ]
        # AGY also stores attached input media in .tempmediaStorage. When the
        # output name is known, prefer the generated artifact's normalized stem
        # (for example output_candidate_v01_a01_png_*.jpg) so an input preview
        # is never mistaken for the render result.
        preferred_token = re.sub(r"[^A-Za-z0-9]+", "_", Path(preferred_name).stem).strip("_").lower()
        if preferred_token:
            matching = [path for path in candidates if preferred_token in path.stem.lower()]
            if matching:
                candidates = matching
            else:
                direct = [path for path in candidates if path.parent == folder]
                if direct:
                    candidates = direct
        return sorted(candidates, key=lambda path: path.stat().st_mtime_ns, reverse=True)
    except OSError:
        return []


def _conversation_ids(value) -> list[str]:
    """Collect AGY conversation IDs from any stream-json nesting level."""
    ids: list[str] = []
    seen: set[str] = set()

    def add(raw) -> None:
        text = str(raw or "").strip()
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{7,127}", text) and text not in seen:
            seen.add(text)
            ids.append(text)

    def walk(obj) -> None:
        if isinstance(obj, dict):
            for key, item in obj.items():
                key_text = str(key).lower().replace("-", "_")
                if key_text in {"conversation_id", "conversationid", "cascade_id"}:
                    if isinstance(item, str):
                        add(item)
                walk(item)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                walk(item)

    walk(value)
    return ids


IMAGE_SUBAGENT = "image-generator"


def _image_subagents(step) -> list[dict]:
    """The built-in ``image-generator`` entries of an ``invoke_subagent`` step (AGY CLI ≥ 1.2.16)."""
    if not isinstance(step, dict):
        return []
    if str(step.get("tool_name") or "").strip().lower() != "invoke_subagent":
        return []
    info = step.get("subagent_info")
    subagents = info.get("subagents") if isinstance(info, dict) else None
    out = []
    for sub in subagents if isinstance(subagents, list) else []:
        if isinstance(sub, dict) and str(sub.get("type_name") or "").strip().lower() == IMAGE_SUBAGENT:
            out.append(sub)
    return out


def _generate_image_tool_info(events: list[dict]) -> tuple[list[dict], bool]:
    """Return image-generation tool payloads and whether AGY sent an image request.

    Two shapes count: the direct ``generate_image`` tool (AGY < 1.2.16), and since 1.2.16 an
    ``invoke_subagent`` step that starts the built-in ``image-generator`` subagent (its child
    conversation id is in ``subagent_info.subagents[]``).  Other subagent types never count."""
    tool_steps: dict[tuple[str, object], dict] = {}
    tool_called = False

    def walk_for_call(value) -> None:
        nonlocal tool_called
        if isinstance(value, dict):
            name = value.get("tool_name") or value.get("name")
            if isinstance(name, str) and name.strip().lower() == "generate_image":
                tool_called = True
            if _image_subagents(value):
                tool_called = True
            for item in value.values():
                walk_for_call(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk_for_call(item)

    for event in events:
        walk_for_call(event)
        step = event.get("step_update") if event.get("event") == "step_update" else None
        if not isinstance(step, dict):
            continue
        if step.get("step_type") == "subagent" and _image_subagents(step):
            tool_called = True   # the image request is out: never send a second one automatically
            key = (str(step.get("conversation_id", "")), step.get("step_index", len(tool_steps)))
            tool_steps[key] = step
            continue
        if step.get("step_type") != "tool":
            continue
        if str(step.get("tool_name") or "").strip().lower() != "generate_image":
            continue
        info = step.get("tool_info")
        if isinstance(info, dict):
            key = (str(step.get("conversation_id", "")), step.get("step_index", len(tool_steps)))
            tool_steps[key] = info
    return list(tool_steps.values()), tool_called


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
    """Generate/edit one image through Antigravity CLI's built-in generate_image tool.

    The CLI agent owns the Google OAuth session. Wondful only stages local images into the
    isolated render workspace and asks the agent to call generate_image with ImagePaths.
    """
    cli = discover_antigravity_cli(explicit_path)
    if not cli:
        raise AntigravityNotInstalledError("未找到 Antigravity CLI（agy）。请先安装官方 Antigravity CLI。")

    workdir = Path(cwd).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    target = Path(output_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    # Blender launched from Finder can have an empty HOME even though the AGY
    # child receives a repaired runtime environment. Use that same resolved
    # home when looking for AGY's conversation brain artifacts.
    brain_roots: list[Path] = []
    for home_text in (_runtime_env(cli).get("HOME", ""), str(Path.home())):
        if not home_text:
            continue
        root = Path(home_text).expanduser() / ".gemini" / "antigravity-cli" / "brain"
        try:
            key = str(root.resolve())
        except OSError:
            key = str(root)
        if key not in {str(item) for item in brain_roots}:
            brain_roots.append(root)
    if target.exists():
        try:
            target.unlink()
        except OSError:
            pass

    staged = workdir / "antigravity_render_inputs"
    staged.mkdir(parents=True, exist_ok=True)
    # A render session is isolated, but remove stale staged bitmaps if a failed retry reused it.
    for old in staged.iterdir():
        try:
            if old.is_file():
                old.unlink()
        except OSError:
            pass

    ordered_sources: list[tuple[str, str]] = []
    for idx, source in enumerate(reference_paths, start=1):
        if not source or not Path(source).expanduser().is_file():
            raise AntigravityError(f"Reference {idx} 已不存在，请重新选择参考图。")
        ordered_sources.append((f"Reference {idx}", source))
    for label, source in (("Edit Base", edit_base_path), ("Edit Mask", edit_mask_path)):
        if source and not Path(source).expanduser().is_file():
            raise AntigravityError(f"{label} 已不存在，请重新捕获结构图。")
    if edit_base_path and Path(edit_base_path).expanduser().is_file():
        ordered_sources.append(("Edit Base", edit_base_path))
    if edit_mask_path and Path(edit_mask_path).expanduser().is_file():
        ordered_sources.append(("Edit Mask", edit_mask_path))

    staged_refs: list[str] = []
    semantic_lines: list[str] = []
    excluded: set[Path] = set()
    staged_by_source = {}
    for index, (label, source) in enumerate(ordered_sources, start=1):
        src = Path(source).expanduser().resolve()
        excluded.add(src)
        # Reference slots remain stable even when the same image has two roles.
        # Only duplicate Edit Base / Edit Mask attachments are aliased.
        if src in staged_by_source:
            semantic_lines.append(f"- {label}: {staged_by_source[src]}（复用已有附件）")
            continue
        suffix = src.suffix.lower() or ".png"
        dest = staged / f"input_{index:02d}{suffix}"
        shutil.copy2(src, dest)
        absolute = str(dest.resolve())
        staged_by_source[src] = absolute
        staged_refs.append(absolute)
        semantic_lines.append(f"- {label}: {absolute}")
        excluded.add(dest.resolve())

    from .reference_bundle import MAX_ANTIGRAVITY_IMAGE_PATHS, check_image_capacity
    check_image_capacity(staged_refs, limit=MAX_ANTIGRAVITY_IMAGE_PATHS)
    # A bundled prompt names source paths; AGY uses staged paths. Do not let the
    # agent attach both copies or re-add the original cells of a sheet.
    for src, rel in sorted(staged_by_source.items(), key=lambda item: len(str(item[0])), reverse=True):
        prompt = prompt.replace(str(src), rel)

    structure_note = ""
    if structure_control and structure_control.get("enabled"):
        packet_roles = list(structure_control.get("generation_reference_roles", []))
        role_lines = [f"- Structure source role: {role}" for role in packet_roles]
        structure_note = f"""
Blender Structure Lock {structure_control.get('packet_version', '2.x')}：
- Camera Base 决定整张 Camera / Composition / Perspective。
- product_bbox_normalized = {structure_control.get('bbox_normalized', [])}
- product_center_normalized = {structure_control.get('center_normalized', [])}
- wheel_points_normalized = {structure_control.get('wheel_points_normalized', [])}
- native data passes = {bool(structure_control.get('native_passes', False))}
{chr(10).join(role_lines) if role_lines else '- 当前没有额外 Structure Packet 图片。'}
Mask/Depth/Normal/Silhouette/Part-ID 都是确定性控制数据，不是最终颜色、线稿或材质。
"""

    image_paths_literal = "[" + ", ".join(repr(x) for x in staged_refs) + "]"
    task = f"""
你是 Wondful AI 渲染器的最终图像生成执行器。当前 Provider 是 Antigravity。

【生图方式】优先用 `invoke_subagent` 调用 Antigravity 内置的 `image-generator` 子代理来生成或编辑最终图片，并等待它完成、拿到结果。
- 只生成 **恰好一张** 图片：不要为了挑质量而重试，也不要生成多张候选。
- 如果当前 CLI 版本没有 `image-generator` 子代理（较旧版本），可以直接调用内置的 `generate_image` 工具一次。
- 插件已经确认输入路径存在，生图工具会自行读取图像；不需要先列出、查看或检查文件。
- 不要用 Python、SVG、HTML、Blender render、外部 API 或 shell 命令伪造图像。
- 把下面所有图像路径、ImageName 和完整的生成要求（含 Structure Lock 约束）原样交给 image-generator 子代理（或 generate_image 的 Prompt、ImageName、ImagePaths）。

输入图像：
{chr(10).join(semantic_lines) if semantic_lines else '- 无额外参考图'}

调用要求：
- ImageName = {target.name!r}
- ImagePaths = {image_paths_literal}
- ImagePaths 只允许上述路径，不要追加图集内部的原始图像，也不要重复添加 Edit Base。
- Prompt 必须完整包含下面的最终生成要求与 Structure Lock 约束。

核心规则：Blender 负责 Structure，Antigravity 生图（image-generator / generate_image）只负责 Appearance。
Reference 1（若存在）是 Blender Camera Base，是原位编辑底图/构图基准；必须保持 Camera / Composition / Geometry / Position / Scale / Rotation / Perspective / Spatial Relationship。
Reference 1 之后若存在 Structure Packet 图片，它们按 Mask / Depth / Normal / Silhouette / Part-ID 分工提供确定性约束；产品图只补充身份与造型识别；环境／风格图决定产品受光；人物图只补充人物身份，均不能改写 Blender 构图。

{REFERENCE_LIGHTING_POLICY}
{structure_note}

如果存在 Edit Base：必须使用本轮指定的图像编辑底图。它与 Reference 1 相同时，直接从原始白模重渲染；否则在候选结果上按修复范围纠偏。不能沿用其他轮次的底图或错误视角。
如果存在 Edit Mask：只在 Mask 对应的产品区域纠正位置、尺度、轮廓、接地点和视角，外部区域尽量不变。

最终生成要求：
{prompt.strip()}

禁止：zoom、pan、crop、reframe、换机位、改变焦段观感、改变主体数量、重新安排产品位置。
完成后请确保真正产生一个位图文件。不要只回复文字描述。
成功时回复 `WONDFUL_ANTIGRAVITY_IMAGEGEN_OK`，并在同一条回复里写出生成图片的绝对路径；如果 image-generator 与 generate_image 都不可用/被策略拒绝/失败，则以 `WONDFUL_ANTIGRAVITY_IMAGEGEN_UNAVAILABLE:` 开头说明原因。
""".strip()

    args = [
        "--output-format", "stream-json",
        "--print-timeout", ("24h" if int(timeout) <= 0 else f"{max(30, int(timeout) - 15)}s"),
    ]
    if model.strip():
        args += ["--model", model.strip()]
    tracker = ArtifactTracker(workdir, excluded)
    # Conversation ids of the most recent AGY turn. Shared with
    # accept_candidate_paths(); 3.1.2-3.1.5 referenced a name local to
    # inspect_result() there, raising NameError after a successful image.
    conversation_ids: list[str] = []

    def run_task(task_text: str) -> subprocess.CompletedProcess:
        invoke_args = list(args) + ["--prompt", task_text]
        return _run(cli, invoke_args, cwd=str(workdir),
                    timeout=(None if int(timeout) <= 0 else int(timeout)))

    def inspect_result(process: subprocess.CompletedProcess):
        events = list(_json_objects(process.stdout or ""))
        final_result = _parse_json_output(process.stdout or "")
        tool_infos, tool_called = _generate_image_tool_info(events)
        status = str(final_result.get("status") or "").upper()
        response = str(final_result.get("response") or "").strip()
        error = str(final_result.get("error") or "").strip()
        # Only generated tool outputs, complete stream events and the final
        # response may nominate artifacts. The tracker rejects echoed inputs or
        # images that predate this invocation.
        candidate_objects = tool_infos + events + [response]
        candidates = [target, workdir / target.name]
        for obj in candidate_objects:
            candidates.extend(_collect_image_candidates(obj, workdir))
        turn_ids = _conversation_ids(events)
        turn_ids.extend(x for x in _conversation_ids(final_result) if x not in turn_ids)
        conversation_ids[:] = turn_ids
        for conversation_id in turn_ids:
            for brain_root in brain_roots:
                candidates.extend(_brain_image_candidates(conversation_id, brain_root, target.name))
        newest = tracker.newest()
        if newest is not None:
            candidates.append(newest)
        return events, final_result, tool_infos, tool_called, status, response, error, candidates

    proc = run_task(task)
    (events, final_result, tool_infos, tool_called, status, response, error,
     candidates) = inspect_result(proc)
    combined = _combined_output(proc)
    if proc.returncode != 0 or (status and status != "SUCCESS") or error:
        detail = error or combined
        if _looks_like_auth_error(detail):
            raise AntigravityNotLoggedInError(f"Antigravity Google 登录不可用或已失效。{detail}".strip())
        capacity_message = _capacity_error_message(detail)
        if capacity_message:
            raise AntigravityImageGenerationUnavailable(capacity_message)
        raise AntigravityError(f"Antigravity 生图执行失败（exit={proc.returncode}）。{detail}".strip())

    def accept_candidate_paths(paths):
        for candidate in paths:
            if tracker.accepts(candidate):
                try:
                    if candidate.resolve() != target:
                        shutil.copy2(candidate, target)
                    return {
                        "engine": "antigravity_oauth_generate_image", "status": "ok",
                        "message": response, "source": str(candidate),
                        "tool_calls": len(tool_infos), "attached_images": len(staged_refs),
                        "image_tool": ("image-generator" if any(_image_subagents(i) for i in tool_infos)
                                       else "generate_image"),
                        "conversation_id": conversation_ids[-1] if conversation_ids else "",
                    }
                except OSError:
                    continue
        return None

    accepted = accept_candidate_paths(candidates)
    # A model may spend its first turn inspecting the workspace and then hit a
    # non-interactive confirmation wall before it reaches generate_image. That
    # turn has not consumed an image request, so one concise retry is safe.
    # Never retry after a generate_image call or once the image-generator
    # subagent has started (AGY ≥ 1.2.16): doing so could
    # double-charge a request whose artifact was merely stored elsewhere.
    if accepted is None and not tool_called:
        retry_task = (
            "立即执行生图：优先用 invoke_subagent 调用内置的 image-generator 子代理并等待它完成；"
            "旧版本没有该子代理时直接调用 generate_image 一次。只生成一张图，不要重试或生成候选。"
            "输入路径已由插件确认存在，不需要先列出或查看文件。成功时回复成功标记和生成图片的绝对路径。\n\n"
            + task
        )
        retry_proc = run_task(retry_task)
        (retry_events, retry_result, retry_infos, retry_called, retry_status, retry_response,
         retry_error, retry_candidates) = inspect_result(retry_proc)
        retry_combined = _combined_output(retry_proc)
        if retry_proc.returncode != 0 or (retry_status and retry_status != "SUCCESS") or retry_error:
            detail = retry_error or retry_combined
            if _looks_like_auth_error(detail):
                raise AntigravityNotLoggedInError(f"Antigravity Google 登录不可用或已失效。{detail}".strip())
            capacity_message = _capacity_error_message(detail)
            if capacity_message:
                raise AntigravityImageGenerationUnavailable(capacity_message)
            raise AntigravityError(f"Antigravity 生图执行失败（重试，exit={retry_proc.returncode}）。{detail}".strip())
        # Prefer the retry's response and tool count for diagnostics/results.
        events, final_result, tool_infos, tool_called, status, response, error = (
            retry_events, retry_result, retry_infos, retry_called,
            retry_status, retry_response, retry_error
        )
        accepted = accept_candidate_paths(retry_candidates)
        combined = retry_combined
        proc = retry_proc
    if accepted is not None:
        return accepted

    denial_detail = (proc.stderr or "").strip()
    capacity_message = _capacity_error_message(response or denial_detail)
    if capacity_message:
        raise AntigravityImageGenerationUnavailable(capacity_message)
    if "WONDFUL_ANTIGRAVITY_IMAGEGEN_UNAVAILABLE" in response or "generate_image" in denial_detail.lower():
        reason = response.split("WONDFUL_ANTIGRAVITY_IMAGEGEN_UNAVAILABLE:", 1)[-1].strip() if "WONDFUL_ANTIGRAVITY_IMAGEGEN_UNAVAILABLE:" in response else denial_detail
        raise AntigravityImageGenerationUnavailable(
            "Antigravity generate_image 当前不可用或被权限/区域/账号策略拒绝。" + (f" {reason[:500]}" if reason else "")
        )
    if not tool_called:
        detail = "本次 Agent turn 未调用 image-generator 子代理或 generate_image；插件已自动重试一次仍未得到位图。"
    else:
        detail = "本次已发出生图请求（image-generator 子代理或 generate_image），但 AGY 没有返回可读取的位图文件；为避免重复扣费未自动再次请求。"
    raise AntigravityImageGenerationUnavailable(
        "Antigravity 已完成 Agent turn，但没有找到 generate_image 产生的可读取位图。 "
        + detail + (f" 返回：{response[:500]}" if response else "")
    )


def polish_prompt(
    *,
    system_prompt: str,
    user_text: str,
    image_paths: Iterable[str],
    cwd: str,
    explicit_path: str = "",
    model: str = "",
    timeout: int = 240,
) -> str:
    prompt = (
        system_prompt.strip()
        + "\n\n--- 当前任务 ---\n"
        + user_text.strip()
        + "\n\n只输出最终可直接用于图像生成的提示词正文。不要解释分析过程，不要输出 Markdown 标题或代码块。请组织成连续自然的一段正文，不要用空行分段，不要输出编号列表。"
    )
    text = _exec_text(
        explicit_path=explicit_path,
        prompt=prompt,
        image_paths=image_paths,
        model=model,
        cwd=cwd,
        timeout=timeout,
    )
    if not text:
        raise AntigravityError("Antigravity 未返回润色后的提示词。")
    return text.strip()


def _extract_json_dict(text: str) -> dict:
    raw = (text or "").strip()
    if not raw:
        raise AntigravityError("Antigravity 构图校验没有返回内容。")
    try:
        direct = json.loads(raw)
        if isinstance(direct, dict):
            return direct
    except Exception:
        pass
    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        raise AntigravityError(f"Antigravity 构图校验未返回 JSON：{raw[:300]}")
    try:
        data = json.loads(match.group(0))
    except Exception as exc:
        raise AntigravityError(f"无法解析 Antigravity 构图校验 JSON：{raw[:300]}") from exc
    if not isinstance(data, dict):
        raise AntigravityError("Antigravity 构图校验结果不是 JSON 对象。")
    return data


def audit_alignment(
    *,
    camera_reference: str,
    generated_image: str,
    cwd: str,
    explicit_path: str = "",
    model: str = "",
    timeout: int = 180,
    tag: str = "",
    structure_control: dict | None = None,
    structure_guide_path: str = "",
) -> dict:
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
    numeric_targets = ""
    if structure_enabled:
        numeric_targets = f"""
Blender 精确投影目标（0~1 归一化坐标，原点左上）：
- target_bbox [left, top, right, bottom] = {bbox}
- target_center [x, y] = {center}
- target_wheel_anchors = {wheels}
这些值来自真实 Blender Camera 投影，优先于视觉猜测。
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

如果有数值锚点，correction 必须尽量量化成画宽/画高百分比，不要只说“更接近参考图”。

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
  "correction": "给下一次 image edit 用的一句中文纠偏指令，具体说明移动方向/百分比、缩放百分比、视角或轮心如何恢复；若已经高度对齐则写保持当前构图"
}}
observed_bbox 必须独立观察 Reference 2 中主体可见边界，坐标以其整幅画布归一化、原点左上，不包含影子，不要照抄 Blender 目标数值。不确定时返回 null，bbox_confidence 设为 0。该框只是视觉估计，不能声称是逐像素测量。轮部件锚点来自命名 Mesh 的包围盒中心估计，仅辅助观察，不能当作精确轮轴。
评分要严格：明显的产品位置、尺度或视角偏差不应超过 80 分。
""".strip()
    images = [camera_reference, generated_image]
    if structure_guide_path and Path(structure_guide_path).exists():
        images.append(structure_guide_path)
    text = _exec_text(
        explicit_path=explicit_path,
        prompt=prompt,
        image_paths=images,
        model=model,
        cwd=cwd,
        timeout=timeout,
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
