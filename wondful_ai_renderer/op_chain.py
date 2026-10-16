"""1.2.2: start a follow-up modal operator (「生成」 → prompt analysis → AI render) robustly.

Pure Python (no bpy) so it can be unit tested; operators._schedule_operator wires it to
bpy.app.timers, bpy.context.temp_override and the operator call.

Before 1.2.2 the chained call ran inside a timer that ignored the operator's result and
let exceptions escape (Blender prints those to the system console, which Windows hides),
so any precheck that returned CANCELLED, a poll/context error, or a missing 3D View left
the panel idle with no message and the user had to click 渲染 by hand.
"""
import time

STARTED = {"RUNNING_MODAL", "FINISHED"}


class OperatorRefused(RuntimeError):
    """The operator ran and refused (it reported an ERROR, which bpy.ops raises as
    RuntimeError).  Not a context problem: trying other contexts would only repeat it."""


def candidate_contexts(windows, scene=None):
    """Context overrides to try, best first.

    Windows showing ``scene`` first; in each, every 3D View with its sidebar (UI) region,
    then its main (WINDOW) region; last, the bare window (the operator then searches the
    window's screen itself and reports a clear error when there is no 3D View)."""
    windows = [w for w in (windows or []) if getattr(w, "screen", None) is not None]
    windows.sort(key=lambda w: 0 if scene is not None and getattr(w, "scene", None) == scene else 1)
    out = []
    for window in windows:
        for area in list(getattr(window.screen, "areas", []) or []):
            if getattr(area, "type", "") != "VIEW_3D":
                continue
            regions = list(getattr(area, "regions", []) or [])
            for rtype in ("UI", "WINDOW"):
                region = next((r for r in regions if getattr(r, "type", "") == rtype), None)
                if region is not None:
                    out.append({"window": window, "area": area, "region": region})
    for window in windows:
        out.append({"window": window})
    return out


class ChainRunner:
    """Timer callback: wait for the task lock, then start the operator in the first
    context that accepts it; retry transient context errors; never stop silently."""

    def __init__(self, *, invoke, contexts, locked, on_fail, on_started=None, last_report=None,
                 lock_wait=900.0, retries=8, retry_interval=0.5, poll_interval=0.3, clock=time.monotonic):
        self.invoke = invoke              # ctx dict -> operator result set (may raise)
        self.contexts = contexts          # () -> list of ctx dicts
        self.locked = locked              # () -> bool (another task still running)
        self.on_fail = on_fail            # (message) -> None
        self.on_started = on_started or (lambda ctx, result: None)
        self.last_report = last_report or (lambda: "")
        self.lock_wait, self.retries = float(lock_wait), int(retries)
        self.retry_interval, self.poll_interval = float(retry_interval), float(poll_interval)
        self.clock = clock
        self.started_at = clock()
        self.attempts = 0
        self.errors = []
        self.done = False
        self.outcome = ""

    def _fail(self, message):
        self.done, self.outcome = True, "failed"
        try:
            self.on_fail(message)
        except Exception as exc:  # pragma: no cover - last resort
            print(f"[Wondful] 自动开始生图失败，且无法显示错误：{exc}；原因：{message}")
        return None

    def step(self):
        if self.done:
            return None
        try:
            if self.locked():
                if self.clock() - self.started_at > self.lock_wait:
                    return self._fail("上一个后台任务一直没有结束")
                return self.poll_interval
            self.attempts += 1
            contexts = list(self.contexts() or [])
            if not contexts:
                self.errors.append("没有可用的 Blender 窗口")
            for ctx in contexts:
                try:
                    result = set(self.invoke(ctx) or ())
                except OperatorRefused as exc:
                    return self._fail(str(exc) or self.last_report() or "生图前检查未通过")
                except Exception as exc:  # poll() failed / context incorrect: try the next one
                    self.errors.append(str(exc).replace("Error: ", "").strip() or type(exc).__name__)
                    continue
                if result & STARTED:
                    self.done, self.outcome = True, "started"
                    self.on_started(ctx, result)
                    return None
                # The operator ran and refused (a precheck). Retrying elsewhere would only repeat it.
                return self._fail(self.last_report() or "生图前检查未通过")
            if self.attempts < self.retries:
                return self.retry_interval
            return self._fail(self.errors[-1] if self.errors else "无法启动")
        except Exception as exc:
            return self._fail(f"{type(exc).__name__}: {exc}")
