"""Offline checks for scene-bound render chaining."""
import contextlib
import sys
import unittest
from types import SimpleNamespace as NS

import test_two_workflows as T


class ScheduleSceneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        T.setUpModule()
        cls._missing=object()
        cls._saved={name:(sys.modules.get(T.PACKAGE+'.'+name,cls._missing),
                          getattr(T.package,name,cls._missing)) for name in ('op_chain','layered_output')}
        T.load('op_chain')
        T.stub('layered_output', RUNNER=NS(pending=0))

    @classmethod
    def tearDownClass(cls):
        for name,(module,attribute) in cls._saved.items():
            if module is cls._missing:
                sys.modules.pop(T.PACKAGE+'.'+name,None)
            else:
                sys.modules[T.PACKAGE+'.'+name]=module
            if attribute is cls._missing:
                if hasattr(T.package,name):
                    delattr(T.package,name)
            else:
                setattr(T.package,name,attribute)
        T.tearDownModule()

    def setUp(self):
        self.origin = NS(name='Scene', wondful_ai=NS(status='IDLE', last_error='', pending_generate=True, views_plan='plan'))
        self.other = NS(name='Other', wondful_ai=NS(status='IDLE', last_error='', pending_generate=True, views_plan='other'))
        self.calls = []
        self.registered = []
        self.window = NS(scene=self.other, screen=NS(areas=[]))
        self.context = NS(scene=self.other, window_manager=NS(windows=[self.window]))
        @contextlib.contextmanager
        def override(**kwargs):
            old=self.context.scene
            self.context.scene=kwargs['window'].scene
            try:
                yield
            finally:
                self.context.scene=old
        self.context.temp_override=override
        self.bpy = NS(context=self.context, data=NS(scenes=[self.origin,self.other]),
                      ops=NS(wondful=NS(ai_render=lambda *a,**k:self.calls.append(self.context.scene) or {'RUNNING_MODAL'})),
                      app=NS(timers=NS(register=lambda fn,**kw:self.registered.append(fn))))
        self.ns = T.function_namespace('_schedule_operator', '_scene_token')
        self.ns.update(bpy=self.bpy, _LAST_REPORT={}, _SOURCE_POLISH_APPROVAL={}, _MULTIVIEW_POLISH_APPROVAL={},
                       _ACTIVE_LOCK=NS(locked=lambda:False), _redraw_all=lambda:None,
                       _status=lambda p,status,error='':(setattr(p,'status',status),setattr(p,'last_error',error)))

    def test_window_switched_to_another_scene_never_invokes_render(self):
        runner=self.ns['_schedule_operator']('wondful.ai_render',self.origin)
        self.assertIsNone(runner.step())
        self.assertEqual(runner.outcome,'failed')
        self.assertEqual(self.calls,[])
        self.assertEqual(self.origin.wondful_ai.status,'ERROR')
        self.assertIn('原始场景',self.origin.wondful_ai.last_error)
        self.assertEqual(self.other.wondful_ai.status,'IDLE')

    def test_same_name_replacement_never_becomes_target_scene(self):
        replacement=NS(name='Scene',wondful_ai=self.other.wondful_ai)
        self.bpy.data.scenes=[replacement]
        self.window.scene=replacement
        runner=self.ns['_schedule_operator']('wondful.ai_render',self.origin)
        self.assertIsNone(runner.step())
        self.assertEqual(self.calls,[])
        self.assertEqual(self.other.wondful_ai.status,'IDLE')

    def test_matching_window_runs_original_scene_only(self):
        original_window=NS(scene=self.origin,screen=NS(areas=[]))
        self.context.window_manager.windows.append(original_window)
        runner=self.ns['_schedule_operator']('wondful.ai_render',self.origin)
        self.assertIsNone(runner.step())
        self.assertEqual(runner.outcome,'started')
        self.assertEqual(self.calls,[self.origin])
        self.assertIs(self.window.scene,self.other)

    def test_renamed_original_scene_stays_the_same_target(self):
        self.window.scene=self.origin
        runner=self.ns['_schedule_operator']('wondful.ai_render',self.origin)
        self.origin.name='renamed after polish'
        self.assertIsNone(runner.step())
        self.assertEqual(self.calls,[self.origin])

    def test_override_with_wrong_context_scene_is_rejected(self):
        self.window.scene=self.origin
        @contextlib.contextmanager
        def wrong_override(**kwargs):
            yield
        self.context.temp_override=wrong_override
        runner=self.ns['_schedule_operator']('wondful.ai_render',self.origin)
        self.assertIsNone(runner.step())
        self.assertEqual(runner.outcome,'failed')
        self.assertEqual(self.calls,[])
        self.assertIn('原始场景',self.origin.wondful_ai.last_error)

    def test_window_changed_after_context_selection_is_rejected(self):
        self.window.scene=self.origin
        runner=self.ns['_schedule_operator']('wondful.ai_render',self.origin)
        contexts=runner.contexts()
        self.window.scene=self.other
        with self.assertRaisesRegex(T.package.op_chain.OperatorRefused,'原始场景'):
            runner.invoke(contexts[0])
        self.assertEqual(self.calls,[])


if __name__ == '__main__':
    unittest.main()
