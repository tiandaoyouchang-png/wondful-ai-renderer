"""Monotonic wall-clock budget shared by retries and provider fallback."""
import time


class RequestDeadline:
    def __init__(self, timeout, clock=None):
        self.clock = clock or time.monotonic
        seconds = float(timeout or 0)
        self.end = self.clock() + seconds if seconds > 0 else None

    def remaining(self):
        if self.end is None:
            return None
        left = self.end - self.clock()
        if left <= 0:
            raise TimeoutError("图像请求已达到总超时，已停止重试和代理回退")
        return left
