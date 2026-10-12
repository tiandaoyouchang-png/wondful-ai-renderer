"""Discover runtime model IDs through official local CLI interfaces, without an AI turn."""
import json
import re
import subprocess
import time
from pathlib import Path

from .cli_runtime import runtime_env
from .cli_transport import ManagedProcess, safe_diagnostic, strip_ansi


def normalize_models(records):
    models = {}
    for row in records:
        if isinstance(row, str):
            row = {"id": row}
        if not isinstance(row, dict) or row.get("hidden"):
            continue
        slug = str(row.get("model") or row.get("slug") or row.get("id") or "").strip()
        if not slug or slug.startswith("-") or re.search(r"\s|[\x00-\x1f\x7f]", slug):
            continue
        models[slug] = {"id": slug,
                        "label": str(row.get("displayName") or row.get("name") or slug),
                        "description": str(row.get("description") or ""),
                        "default": bool(row.get("isDefault") or row.get("default"))}
    return list(models.values())


def parse_agy_models(text):
    text = strip_ansi(text)
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        data = None
    if isinstance(data, dict):
        data = data.get("models", data.get("data", []))
    if isinstance(data, list):
        return normalize_models(data)
    rows = []
    for line in text.splitlines():
        # Documented `agy models`: slug followed by its display name. Accept
        # table borders / current-selection markers, never fabricate model IDs.
        match = re.match(r"^[\s│|>*•✓]*([a-z][a-z0-9_.]*(?:[-/][a-z0-9_.]+)+)\s+(.+?)\s*$", line)
        if match:
            slug, label = match.groups()
            rows.append({"id": slug, "displayName": label.rstrip(" │|"),
                         "isDefault": bool(re.search(r"\((?:default|current)\)", label, re.I))})
    return normalize_models(rows)


def agy_models(cli, *, cwd, timeout=30):
    proc = subprocess.run([cli, "models"], cwd=cwd, env=runtime_env(cli),
                          stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True, encoding="utf-8", errors="replace", timeout=timeout)
    if proc.returncode:
        raise RuntimeError("AGY 模型列表读取失败：" + safe_diagnostic(proc.stderr or proc.stdout))
    models = parse_agy_models(proc.stdout)
    if not models:
        raise RuntimeError("当前 AGY 未返回可解析的模型 ID。请更新官方 CLI，或在模型框输入官方模型 ID。")
    return models


def codex_models(cli, *, cwd, timeout=30, process_factory=ManagedProcess):
    """Initialize -> model/list (all pages); never thread/start or turn/start."""
    process = process_factory(cli, ["app-server"], cwd=cwd, env=runtime_env(cli), use_tty=False)
    deadline = time.monotonic() + timeout
    pending, buffer = {}, ""

    def request(request_id, method, params):
        nonlocal buffer
        process.write_line(json.dumps({"id": request_id, "method": method, "params": params}))
        while time.monotonic() < deadline:
            buffer += process.read_available()
            lines = buffer.split("\n")
            buffer = lines.pop()
            for line in lines:
                try:
                    message = json.loads(line)
                except ValueError:
                    continue  # Diagnostic stderr can share this pipe.
                if isinstance(message, dict) and "id" in message and ("result" in message or "error" in message):
                    pending[message["id"]] = message
            if request_id in pending:
                response = pending.pop(request_id)
                if response.get("error"):
                    raise RuntimeError("Codex 模型接口错误：" + safe_diagnostic(str(response["error"])))
                return response.get("result")
            if process.proc.poll() is not None:
                process._reader.join(timeout=.1)
                tail = process.read_available()
                if tail:
                    buffer += tail
                    continue
                raise RuntimeError("Codex app-server 提前退出；请检查 CLI 路径、版本与登录状态。")
            time.sleep(.03)
        raise RuntimeError("读取 Codex 模型列表超时；请检查 CLI 状态后重试。")

    try:
        request(1, "initialize", {"clientInfo": {"name": "wondful_renderer", "title": "Wondful AI Renderer", "version": "1.1.0"}})
        process.write_line(json.dumps({"method": "initialized"}))
        rows, cursor, seen = [], None, set()
        for request_id in range(2, 22):
            params = {"limit": 100}
            if cursor:
                params["cursor"] = cursor
            result = request(request_id, "model/list", params)
            if not isinstance(result, dict) or not isinstance(result.get("data"), list):
                raise RuntimeError("Codex model/list 返回了不兼容的格式，请更新官方 CLI。")
            rows.extend(result["data"])
            cursor = result.get("nextCursor")
            if not cursor:
                models = normalize_models(rows)
                if not models:
                    raise RuntimeError("Codex 当前没有返回可选择的模型。请检查账号与 CLI 配置。")
                return models
            if cursor in seen:
                raise RuntimeError("Codex 模型列表分页异常；未将不完整列表标记为成功。")
            seen.add(cursor)
        raise RuntimeError("Codex 模型列表超过分页上限。")
    finally:
        process.close()


def discover_models(provider, cli, *, cwd):
    if not cli:
        raise RuntimeError("未找到当前服务的 CLI，请先选择 CLI 路径或检测安装。")
    Path(cwd).mkdir(parents=True, exist_ok=True)
    return (agy_models if provider == "antigravity" else codex_models)(cli, cwd=str(cwd))
