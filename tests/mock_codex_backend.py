"""Mock of the Codex Responses backend + OAuth token endpoint (tests only).

Run standalone:  python3 tests/mock_codex_backend.py --port 8765 --log /tmp/mock.jsonl [--mode ok]
Or import and use ``start_server(mode=...)`` from unittest.

POST /backend-api/codex/responses  -> SSE stream with one image_generation_call
POST /oauth/token                  -> refreshed tokens
Every request is recorded (tokens redacted, data URLs shortened).
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import struct
import threading
import time
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def fake_jwt(claims: dict) -> str:
    def seg(obj):
        return base64.urlsafe_b64encode(json.dumps(obj).encode()).rstrip(b"=").decode()
    return f"{seg({'alg': 'none'})}.{seg(claims)}.sig"


def write_fake_auth(path, *, exp_offset=3600, account="acct_test_123", access="ACCESS-SECRET-1"):
    access_token = fake_jwt({"exp": int(time.time()) + exp_offset, "tag": access})
    id_token = fake_jwt({"https://api.openai.com/auth": {"chatgpt_account_id": account}})
    data = {"OPENAI_API_KEY": None, "tokens": {"id_token": id_token, "access_token": access_token,
                                              "refresh_token": "REFRESH-SECRET-1"},
            "last_refresh": "2026-10-10T00:00:00.000Z"}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return access_token


def solid_png(width, height, rgb=(90, 140, 200)):
    row = b"\x00" + bytes(rgb) * width
    raw = row * height

    def chunk(kind, payload):
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def edited_png(first_image_data_url, size):
    """Tint image 1 and resize it to the requested size (needs Pillow; else solid colour)."""
    w, h = (int(v) for v in size.split("x")) if size and size != "auto" else (1024, 1024)
    try:
        from PIL import Image, ImageDraw
        raw = base64.b64decode(first_image_data_url.split(",", 1)[1])
        src = Image.open(io.BytesIO(raw)).convert("RGB").resize((w, h))
        cool = Image.new("RGB", (w, h), (150, 185, 230))
        img = Image.blend(src, cool, 0.3)
        d = ImageDraw.Draw(img)
        d.rectangle((0, 0, 360, 34), fill=(30, 120, 60))
        d.text((10, 10), "MOCK CODEX DIRECT EDIT", fill="white")
        out = io.BytesIO()
        img.save(out, "PNG")
        return out.getvalue()
    except Exception:
        return solid_png(w, h)


# 1.2.3: error bodies. http_401_real and the 400 shape were recorded from the real ChatGPT Codex
# backend on 2026-10-17 (bad token / unknown parameter; no generation spent). The 429 and in-stream
# failure bodies follow the shapes the open-source Codex CLI parses (usage_limit_reached with
# resets_in_seconds; rate_limit_exceeded "Please try again in Ns"); ids/accounts sanitized.
FIXTURES = {
    "http_401_real": b'{"detail":"Could not parse your authentication token. Please try signing in again."}',
    "http_400_real": b'{"detail":"Unsupported parameter: bogus_param_xyz"}',
    "http_429_usage_limit": (b'{"error":{"type":"usage_limit_reached","message":"The usage limit has been reached",'
                             b'"plan_type":"plus","resets_in_seconds":7200}}'),
    "http_429_rate": (b'{"error":{"type":"rate_limit_exceeded","code":"rate_limit_exceeded",'
                      b'"message":"Rate limit reached for requests. Please try again in 1s."}}'),
    "http_400_usage_not_included": (b'{"error":{"type":"usage_not_included","message":'
                                    b'"Image generation is not included in your current plan usage."}}'),
    "sse_failed_usage_limit": ('{"type":"response.failed","response":{"id":"resp_1","status":"failed","error":'
                               '{"code":"usage_limit_reached","message":"You\'ve hit your usage limit. Try again later."}}}'),
    "sse_failed_rate_limit": ('{"type":"response.failed","response":{"id":"resp_1","status":"failed","error":'
                              '{"code":"rate_limit_exceeded","message":"Rate limit reached. Please try again in 0.3s."}}}'),
    "sse_error_token_invalidated": ('{"type":"error","error":{"type":"invalid_request_error","code":"token_invalidated",'
                                    '"message":"Your authentication token has been invalidated. Please try signing in again."}}'),
}


def _shorten(value):
    if isinstance(value, dict):
        return {k: _shorten(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_shorten(v) for v in value]
    if isinstance(value, str) and value.startswith("data:"):
        return value.split(",", 1)[0] + f",<{len(value)}>"
    return value


class State:
    def __init__(self, mode="ok", log_path=""):
        self.mode = mode
        self.log_path = log_path
        self.requests = []
        self.raw_bodies = []
        self.lock = threading.Lock()
        self.response_calls = 0
        self.token_calls = 0
        self.valid_token_tags = {"ACCESS-SECRET-1", "ACCESS-SECRET-2"}

    def record(self, entry):
        with self.lock:
            self.requests.append(entry)
            if self.log_path:
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def make_handler(state: State):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def _send(self, code, body: bytes, ctype="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_h(self, code, body: bytes, headers: dict, ctype="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            for k, v in headers.items():
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            headers = {k: v for k, v in self.headers.items()}
            auth = headers.get("Authorization", "")
            token_tag = ""
            if auth.startswith("Bearer "):
                try:
                    seg = auth[7:].split(".")[1]
                    token_tag = json.loads(base64.urlsafe_b64decode(seg + "=" * (-len(seg) % 4))).get("tag", "")
                except Exception:
                    token_tag = "?"
            red = dict(headers)
            if "Authorization" in red:
                red["Authorization"] = "Bearer <redacted:" + token_tag + ">"
            if self.path == "/_mode":  # test control: switch behaviour at runtime
                state.mode = json.loads(body or b"{}").get("mode", "ok")
                return self._send(200, json.dumps({"mode": state.mode}).encode())
            if self.path.endswith("/oauth/token"):
                state.token_calls += 1
                payload = json.loads(body or b"{}")
                state.record({"path": self.path, "headers": red, "grant": payload.get("grant_type"),
                              "client_id": payload.get("client_id")})
                new_access = fake_jwt({"exp": int(time.time()) + 3600, "tag": "ACCESS-SECRET-2"})
                return self._send(200, json.dumps({"access_token": new_access, "refresh_token": "REFRESH-SECRET-2",
                                                   "id_token": fake_jwt({"https://api.openai.com/auth": {"chatgpt_account_id": "acct_test_123"}})}).encode())
            if not self.path.endswith("/responses"):
                return self._send(404, b'{"detail":"Not Found"}')
            state.response_calls += 1
            payload = json.loads(body or b"{}")
            state.raw_bodies.append(payload)
            state.record({"path": self.path, "headers": red, "payload": _shorten(payload)})
            mode = state.mode
            if mode == "401_then_ok" and token_tag != "ACCESS-SECRET-2":
                return self._send(401, b'{"error":{"message":"token expired"}}')
            if token_tag not in state.valid_token_tags:
                return self._send(401, b'{"error":{"message":"bad token"}}')
            if mode == "404":
                return self._send(404, b'{"detail":"Not Found"}')
            if mode == "500":
                return self._send(500, b'{"error":"boom"}')
            if mode == "400_then_ok" and payload["tools"][0].get("model"):
                return self._send(400, b'{"error":{"message":"Unknown parameter: tools[0].model"}}')
            if mode == "quota":
                return self._send(429, FIXTURES["http_429_usage_limit"])
            # 1.2.3 error-classification fixtures (sanitized; see FIXTURES)
            if mode == "401_always":
                return self._send(401, FIXTURES["http_401_real"])
            if mode == "rate_then_ok" and state.response_calls == 1:
                return self._send_h(429, FIXTURES["http_429_rate"], {"Retry-After": "1"})
            if mode == "rate_long":
                return self._send_h(429, FIXTURES["http_429_rate"], {"Retry-After": "900"})
            if mode == "400_quota_text":
                return self._send(400, FIXTURES["http_400_usage_not_included"])
            if mode in ("stream_quota", "stream_rate_then_ok", "stream_auth") and not (
                    mode == "stream_rate_then_ok" and state.response_calls > 1):
                key = {"stream_quota": "sse_failed_usage_limit", "stream_rate_then_ok": "sse_failed_rate_limit",
                       "stream_auth": "sse_error_token_invalidated"}[mode]
                events = [{"type": "response.created", "response": {"id": "resp_1"}}, json.loads(FIXTURES[key])]
                stream = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()
                return self._send(200, stream, "text/event-stream")
            if mode in ("stream_cut_then_ok", "stream_cut_always") and not (
                    mode == "stream_cut_then_ok" and state.response_calls > 1):
                # headers + the first event, then the connection drops (no Content-Length, close)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()
                ev = {"type": "response.created", "response": {"id": "resp_cut"}}
                self.wfile.write(f"event: response.created\ndata: {json.dumps(ev)}\n\n".encode())
                self.wfile.flush()
                self.close_connection = True
                return
            content = payload["input"][0]["content"]
            first = next(c["image_url"] for c in content if c.get("type") == "input_image")
            png = edited_png(first, payload["tools"][0].get("size", "auto"))
            b64 = base64.b64encode(png).decode()
            item = {"id": "ig_1", "type": "image_generation_call", "status": "completed", "result": b64,
                    "revised_prompt": "mock revised prompt", "output_format": "png"}
            if mode == "failed_event":
                events = [{"type": "response.created", "response": {"id": "resp_1"}},
                          {"type": "response.failed", "response": {"id": "resp_1", "error": {"code": "server_error", "message": "mock failure"}}}]
            elif mode == "json":
                return self._send(200, json.dumps({"id": "resp_json", "output": [item]}, indent=1).encode())
            else:
                events = [{"type": "response.created", "response": {"id": "resp_1"}},
                          {"type": "response.image_generation_call.in_progress", "item_id": "ig_1"},
                          {"type": "response.output_item.done", "item": item},
                          {"type": "response.completed", "response": {"id": "resp_1", "output": []}}]
            stream = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(stream)))
            self.end_headers()
            # write in small pieces to exercise incremental parsing
            for i in range(0, len(stream), 7000):
                self.wfile.write(stream[i:i + 7000])
                self.wfile.flush()
    return Handler


def start_server(mode="ok", log_path="", port=0):
    state = State(mode, log_path)
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(state))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, state


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--log", default="")
    ap.add_argument("--mode", default="ok")
    a = ap.parse_args()
    srv, _st = start_server(a.mode, a.log, a.port)
    print(f"mock codex backend on 127.0.0.1:{srv.server_address[1]} mode={a.mode}", flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass
