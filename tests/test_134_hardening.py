"""PC hardening: negative cases for wall fill, cache identity, deadline and cancellation."""
import copy
import json
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_316_regressions import load_package_module
from test_321_direct_edit import DirectEditTestBase, direct
from test_133_ground_bands import GB, MV, MASTER, ANALYSIS

Budget = load_package_module('request_budget').RequestDeadline
CC = load_package_module('codex_client')


class GroundSafety(unittest.TestCase):
    def test_flat_colour_does_not_create_height(self):
        info = GB.measure(MASTER, analysis=ANALYSIS)
        self.assertTrue(info)
        self.assertFalse(any(GB.raised(b) for b in info['bands']))
        self.assertNotIn('略高于两侧', GB.bands_text(info)[0])

    def test_measured_step_creates_curb(self):
        view = copy.deepcopy(MASTER)
        p = view['pos']
        sel = (p[..., 1] >= -5.3) & (p[..., 1] < -4.8) & (p[..., 3] > 0.5)
        p[sel, 2] = 0.12
        info = GB.measure(view, analysis=ANALYSIS)
        curbs = [b for b in info['bands'] if GB.raised(b)]
        self.assertEqual(len(curbs), 1)
        self.assertTrue(curbs[0]['height_measured'])
        self.assertAlmostEqual(curbs[0]['height'], 0.12, delta=0.02)

    def test_wall_and_ceiling_not_filled_but_curb_face_is(self):
        info = dict(t=[1, 0], n=[0, 1], a0=[0, 0], plane=[0, 0, 0],
                    bands=[dict(kind='main', s0=-5, s1=0, height=0),
                           dict(kind='curb', s0=0, s1=0.4, height=0.12)])
        v = dict(pos=np.array([[[0, -2, 0.02, 1], [0, 0.2, 0.1, 1],
                               [0, -2, 0, 1], [0, -2, 0, 1]]]),
                 nrm=np.array([[[0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, -1]]]),
                 mask=np.zeros((1, 4)))
        ground, *_ = GB.ground_level(v, info)
        self.assertEqual(ground.tolist(), [[False, True, True, False]])


class PoolIdentity(unittest.TestCase):
    def test_key_canonical_and_sensitive(self):
        self.assertEqual(MV.cache_key({'x': 1, 'y': 2}), MV.cache_key({'y': 2, 'x': 1}))
        self.assertNotEqual(MV.cache_key('image', [0, 1]), MV.cache_key('image', [1, 0]))
        self.assertNotEqual(MV.cache_key('old_image', [0, 1]), MV.cache_key('image', [0, 1]))

    def test_old_key_and_missing_pass_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            entry = dict(name='other', geometry='detail', image='i', pos='p', nrm='n', mask='m')
            for key in ('image', 'pos', 'nrm', 'mask'):
                (Path(tmp) / entry[key]).touch()
            data = dict(context_key='current', views=[entry])
            self.assertEqual(MV.reusable_views(data, 'current', tmp, ['master'], 'detail'), [entry])
            self.assertEqual(MV.reusable_views(data, 'changed', tmp, [], 'detail'), [])
            self.assertEqual(MV.reusable_views(dict(views=[entry]), 'current', tmp, [], 'detail'), [])
            self.assertEqual(MV.reusable_views(data, 'current', tmp, [], 'llm'), [])
            self.assertEqual(MV.reusable_views(data, 'current', tmp, ['other'], 'detail'), [])
            (Path(tmp) / 'n').unlink()
            self.assertEqual(MV.reusable_views(data, 'current', tmp, [], 'detail'), [])


class DeadlineTests(unittest.TestCase):
    def test_monotonic_remaining_and_unlimited(self):
        clock = [10.0]
        b = Budget(3, clock=lambda: clock[0])
        clock[0] = 12.5
        self.assertEqual(b.remaining(), 0.5)
        clock[0] = 13
        with self.assertRaises(TimeoutError):
            b.remaining()
        self.assertIsNone(Budget(0).remaining())

    def test_cancel_before_network(self):
        with patch.object(direct.urlrequest, 'urlopen') as request:
            with self.assertRaises(direct.DirectEditError) as exc:
                direct.post_responses('http://localhost', {}, {}, timeout=10, cancel_check=lambda: True)
            self.assertEqual(exc.exception.kind, 'cancelled')
            request.assert_not_called()

    def test_cancel_before_fallback_even_when_direct_disabled(self):
        calls = []
        with self.assertRaisesRegex(RuntimeError, '取消'):
            direct.generate_with_fallback(dict(timeout=10), fallback=lambda **k: calls.append(k), state=dict(disabled='offline'), size='32x32', cancel_check=lambda: True)
        self.assertEqual(calls, [])

    def test_fractional_exec_timeout_kept(self):
        with patch.object(CC.subprocess, 'run', return_value=None) as run:
            CC._run('/fake', [], timeout=0.25)
        self.assertEqual(run.call_args.kwargs['timeout'], 0.25)

    def test_exec_stdin_and_cancellation(self):
        transport = load_package_module('cli_transport')
        transport.reset_cancellation()
        proc = CC._run(sys.executable, ['-c', 'import sys; print(sys.stdin.read())'],
                       input_text='中文 prompt', timeout=3)
        self.assertIn('中文 prompt', proc.stdout)
        timer = threading.Timer(0.15, transport.request_cancellation)
        timer.start()
        start = time.monotonic()
        try:
            with self.assertRaisesRegex(CC.CodexError, '取消'):
                CC._run(sys.executable, ['-c', 'import time; time.sleep(30)'],
                        input_text='prompt', timeout=3)
            self.assertLess(time.monotonic() - start, 2)
        finally:
            timer.cancel()
            transport.reset_cancellation()

    def test_model_fallback_shares_deadline(self):
        clock = [0.0]
        deadline = Budget(1, clock=lambda: clock[0])
        seen = []
        def run(**kwargs):
            seen.append(kwargs['timeout'])
            clock[0] += 0.6
            if len(seen) == 1:
                raise CC.CodexModelRejectedError('unsupported model')
            return 'ok'
        with patch.object(CC, 'RequestDeadline', return_value=deadline), \
             patch.object(CC, '_exec_turn_once', side_effect=run):
            self.assertEqual(CC._exec_turn(model='rejected', timeout=1), 'ok')
        self.assertEqual(len(seen), 2)
        self.assertAlmostEqual(seen[1], 0.4)

    def test_trickling_stream_cannot_extend_total_timeout(self):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_POST(self):
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.end_headers()
                try:
                    for _ in range(30):
                        self.wfile.write(b': heartbeat\n\n')
                        self.wfile.flush()
                        time.sleep(0.03)
                except (BrokenPipeError, ConnectionResetError):
                    pass
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        start = time.monotonic()
        try:
            with self.assertRaises(direct.DirectEditError) as exc:
                direct.post_responses(f'http://127.0.0.1:{server.server_port}', {}, {}, timeout=0.2)
            self.assertEqual(exc.exception.kind, 'timeout')
            self.assertLess(time.monotonic() - start, 0.8)
        finally:
            server.shutdown()
            server.server_close()


class ProviderDeadline(DirectEditTestBase):
    def test_retry_consumes_one_deadline(self):
        clock = [0.0]
        limits = []
        def post(*a, timeout, **kw):
            limits.append(timeout)
            clock[0] += 4
            raise direct.DirectEditError('HTTP 400', kind='request')
        with patch.object(direct, 'post_responses', side_effect=post):
            with self.assertRaises(direct.DirectEditError) as exc:
                self.edit(deadline=Budget(7, clock=lambda: clock[0]))
        self.assertEqual(limits, [7, 3])
        self.assertEqual(exc.exception.kind, 'timeout')

    def test_fallback_receives_only_remaining_time(self):
        deadline = Budget(10)
        deadline.remaining = lambda: 3.25
        seen = []
        with patch.object(direct, 'RequestDeadline', return_value=deadline), \
             patch.object(direct, 'edit_image', side_effect=direct.DirectEditError('offline', kind='network')):
            direct.generate_with_fallback(dict(prompt='x', reference_paths=[self.camera], output_path=str(self.tmp/'out.png'), timeout=10), fallback=lambda **kw: seen.append(kw) or {},
                size='32x32', state={})
        self.assertEqual(seen[0]['timeout'], 3.25)

    def test_expiry_stops_fallback(self):
        seen = []
        with patch.object(direct, 'edit_image', side_effect=direct.DirectEditError('总超时', kind='timeout')):
            with self.assertRaises(direct.DirectEditStop):
                direct.generate_with_fallback(dict(reference_paths=[self.camera], output_path=str(self.tmp/'out.png'), timeout=1), fallback=lambda **kw: seen.append(kw), size='32x32', state={})
        self.assertEqual(seen, [])


if __name__ == '__main__':
    unittest.main()
