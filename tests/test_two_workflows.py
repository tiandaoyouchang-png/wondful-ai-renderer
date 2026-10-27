"""Offline regression tests. Blender RNA / UILayout are explicitly simulated.

The implementation under test is real source. These tests do NOT constitute a
native Blender or image-model end-to-end acceptance test.
"""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "wondful_ai_renderer"
NS = types.SimpleNamespace
PACKAGE = "_wondful_two_workflows_test"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(PKG)]
sys.modules[PACKAGE] = package


def load(name):
    full = PACKAGE + "." + name
    spec = importlib.util.spec_from_file_location(full, PKG / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[full] = module
    setattr(package, name, module)
    spec.loader.exec_module(module)
    return module


WF = load("workflow")


class Item:
    def __init__(self, name="item", tags=None, kind="MESH"):
        self.name, self.tags, self.type = name, tags or {}, kind
        self.hide_render = False
        self.hidden = False
        self.data = NS(materials=[])
        self.parent = None
        self.users_collection = []
    def get(self, key, default=None):
        return self.tags.get(key, default)
    def hide_get(self, **kwargs):
        return self.hidden
    def hide_set(self, value, **kwargs):
        self.hidden = value


class Collection(Item):
    def __init__(self, name="collection", objects=(), children=(), tags=None):
        super().__init__(name, tags)
        self.objects = list(objects)
        self.children = list(children)
        self.children_recursive = list(children)
        for child in children:
            self.children_recursive.extend(child.children_recursive)
        self.all_objects = list(objects)
        for child in children:
            self.all_objects.extend(child.all_objects)


class SceneObjects(list):
    def __contains__(self, item):
        return any(o.name == item for o in self) if isinstance(item, str) else super().__contains__(item)


class Operator:
    def report(self, level, message):
        self.last_report = (level, message)


def defaults(class_name):
    tree = ast.parse((PKG / "properties.py").read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    values = {}
    for n in cls.body:
        if not isinstance(n, ast.AnnAssign) or not isinstance(n.annotation, ast.Call):
            continue
        call = n.annotation
        keywords = {k.arg: k.value for k in call.keywords}
        if "default" in keywords:
            values[n.target.id] = ast.literal_eval(keywords["default"])
        elif call.func.id == "CollectionProperty":
            values[n.target.id] = []
        elif call.func.id == "PointerProperty":
            values[n.target.id] = None
        elif call.func.id == "EnumProperty":
            values[n.target.id] = ast.literal_eval(keywords["items"])[0][0]
        else:
            values[n.target.id] = "" if call.func.id == "StringProperty" else False if call.func.id == "BoolProperty" else 0
    return NS(**values)


def props(**kw):
    p = defaults("WONDFUL_AISettings")
    p.product_images = [NS(image=NS(name="Identity",size=(160,90),filepath=__file__,preview_ensure=lambda:None), source_path=__file__, instruction="")]
    p.__dict__.update(kw)
    return p


def context(p=None):
    p = p or props()
    product_obj = Item("YU7")
    car = Collection("Car", [product_obj])
    camera = Item("User camera", kind="CAMERA")
    camera.data = NS(type="PERSP", lens=45.0, sensor_fit="HORIZONTAL", sensor_width=36.0,
                     sensor_height=24.0, shift_x=0.0, shift_y=0.0)
    root = Collection("Scene", [product_obj, camera], [car])
    p.product_collection = car
    prefs = defaults("WONDFUL_AddonPreferences")
    sc = NS(wondful_ai=p, camera=camera, collection=root, objects=SceneObjects(root.all_objects),
            render=NS(resolution_x=1600, resolution_y=900, resolution_percentage=100))
    return NS(scene=sc, view_layer=NS(), region=NS(width=380), active_object=None,
              selected_objects=[], window_manager=NS(wondful_auth_code=""),
              preferences=NS(addons={PACKAGE: NS(preferences=prefs)}, system=NS(ui_scale=1.0)))


def function_namespace(*names, classes=()):
    tree = ast.parse((PKG / "operators.py").read_text(encoding="utf-8"))
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names
                or isinstance(n, ast.ClassDef) and n.name in classes]
    env = {"__package__": PACKAGE, "__name__": PACKAGE + ".operators", "workflow": WF,
           "Operator": Operator, "_BaseAsyncOperator": Operator, "Path": Path,
           "json": json, "hashlib": __import__("hashlib"), "time": __import__("time"),
           "MAX_PRODUCT_REFERENCES": 1, "_ACTIVE_LOCK": NS(locked=lambda: False),
           "StringProperty": lambda **kw: kw.get("default", "")}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(PKG / "operators.py"), "exec"), env)
    return env


# Narrow RNA stand-ins: no Blender, credentials or image-model calls.
bpy = types.ModuleType("bpy")
bpy_types = types.ModuleType("bpy.types")
bpy_types.Operator = Operator
bpy_types.Panel = type("Panel", (), {})
bpy_types.UIList = type("UIList", (), {})
bpy_types.UILayout = type("UILayout", (), {"icon": staticmethod(lambda image: 1)})
bpy_props = types.ModuleType("bpy.props")
bpy_props.FloatProperty = lambda **kw: kw.get("default", 0.0)
bpy.types = bpy_types
bpy.props = bpy_props
bpy.path = NS(abspath=lambda p: str(Path(p).resolve()) if p else "")
bpy.data = NS(objects=[], scenes=[], texts=NS(get=lambda name: None))
bpy.app = NS(translations=NS(pgettext_iface=lambda text: text), version=(4, 3, 2))
_saved_bpy_modules = {}


def setUpModule():
    # Do not pollute imports of other regression modules during discovery.
    for name, module in {"bpy": bpy, "bpy.types": bpy_types, "bpy.props": bpy_props}.items():
        _saved_bpy_modules[name] = sys.modules.get(name)
        sys.modules[name] = module


def tearDownModule():
    for name, module in _saved_bpy_modules.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module
    _saved_bpy_modules.clear()


def stub(name, **items):
    module = types.ModuleType(PACKAGE + "." + name)
    module.__dict__.update(items)
    sys.modules[module.__name__] = module
    setattr(package, name, module)
    return module


MV = stub("multiview", scene_blockout_exists=lambda scene: any(c.get("wondful_scene_blockout") for c in scene.collection.children_recursive),
          checked_views=lambda slots, is_present: [c for c, enabled in slots if c and enabled and is_present(c)])
MS = load("massing")
OPS = function_namespace("_reference_paths", "_reference_instructions", "_sb_reference", "_sb_generation_reference",
                         "_sb_ground_band_line", "generate_needs_polish", "generate_views", "workflow_preflight", "identity_preflight",
                         classes=("WONDFUL_OT_generate", "WONDFUL_OT_ai_render"))
OPS.update(bpy=bpy, image_filepath=lambda image: getattr(image, "filepath", "") if image else "",
           _addon_prefs=lambda c: c.preferences.addons[PACKAGE].preferences,
           _status=lambda p,s,e="": (setattr(p,"status",s),setattr(p,"last_error",e)),
           _LAST_REPORT={}, _analysis_provider=lambda p, pref: {"id":"codex"},
           needs_reference_policy_refresh=lambda p: False,
           prompt_profiles=NS(needs_target_refresh=lambda p,pref: False),
           _product_autofill_key=lambda *a: "",
           _load_reference_image=lambda path:NS(size=(160,90)))


class WorkflowStateTests(unittest.TestCase):
    def test_legacy_scene_defaults_to_environment(self):
        self.assertEqual(WF.mode(NS()), WF.ENVIRONMENT)
    def test_switch_preserves_exact_drafts_and_camera(self):
        p=props(prompt="env draft", scene_brief="snow", last_ai_prompt="AI old")
        c=context(p);camera=c.scene.camera;product=p.product_collection
        p.workflow_mode=WF.REFERENCE;WF.switch_mode(p,c)
        self.assertEqual(p.prompt,"");self.assertEqual(p.scene_brief,"")
        p.prompt="replica draft";p.scene_brief="concrete"
        p.workflow_mode=WF.ENVIRONMENT;WF.switch_mode(p,c)
        self.assertEqual(p.prompt,"env draft");self.assertEqual(p.scene_brief,"snow")
        p.workflow_mode=WF.REFERENCE;WF.switch_mode(p,c)
        self.assertEqual(p.prompt,"replica draft")
        self.assertIs(c.scene.camera,camera);self.assertIs(p.product_collection,product)
    def test_switch_does_not_delete_target_or_references(self):
        p=props(sb_image_path="target.png",style_images=["style"],replica_reviewed=True)
        p.workflow_mode=WF.REFERENCE;WF.switch_mode(p)
        self.assertEqual(p.sb_image_path,"target.png");self.assertEqual(p.style_images,["style"])
        self.assertTrue(p.replica_reviewed)
    def test_busy_modes_are_rejected(self):
        for state in WF.BUSY:
            with self.subTest(state=state):
                p=props(status=state,prompt="keep",workflow_mode=WF.REFERENCE)
                WF.switch_mode(p)
                self.assertEqual(p.workflow_mode,WF.ENVIRONMENT);self.assertEqual(p.prompt,"keep")
    def test_invalid_draft_json_keeps_current_text(self):
        p=props(workflow_drafts="not json",prompt="save me",workflow_mode=WF.REFERENCE)
        WF.switch_mode(p)
        self.assertEqual(json.loads(p.workflow_drafts)[WF.ENVIRONMENT]["prompt"],"save me")
    def test_manual_copy_is_not_rewritten_for_changed_references(self):
        p=props(prompt="my exact brief",last_ai_prompt="old AI",scene_brief="short")
        self.assertEqual(WF.render_text(p,True),"my exact brief")
    def test_stale_ai_copy_is_not_reused(self):
        p=props(prompt="old AI",last_ai_prompt="old AI",scene_brief="current brief")
        self.assertEqual(WF.render_text(p,True),"current brief")
    def test_blank_prompt_has_safe_mode_specific_default(self):
        self.assertIn("当前相机",WF.render_text(props()))
        self.assertIn("已确认",WF.render_text(props(workflow_mode=WF.REFERENCE)))
    def test_generate_always_polishes_even_with_legacy_flags_disabled(self):
        p=props(style_prompt_dirty=True,prompt="",style_images=[1])
        self.assertTrue(OPS["generate_needs_polish"](p,NS()))
    def test_legacy_auto_polish_flag_does_not_change_generate_flow(self):
        self.assertTrue(OPS["generate_needs_polish"](props(auto_polish_on_generate=True),NS()))


class ReferenceAndGuardTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.target=Path(self.tmp.name)/"target.png";self.target.write_bytes(b"target-image-bytes")
    def ready(self):
        return props(workflow_mode=WF.REFERENCE,sb_image_path=str(self.target),
                     sb_built_target_hash=WF.file_hash(self.target),replica_reviewed=True)
    def check(self,p,**kw):
        args=dict(has_camera=True,has_product=True,has_blockout=True,target_path=str(self.target),source_ok=True)
        args.update(kw);return WF.preflight(p,**args)
    def test_environment_needs_no_target_or_blockout(self):
        self.assertEqual(self.check(props(),has_blockout=False,target_path="",source_ok=False),"")
    def test_both_workflows_require_product_and_camera(self):
        for p in (props(),self.ready()):
            self.assertTrue(self.check(p,has_product=False));self.assertTrue(self.check(p,has_camera=False))
    def test_replica_requires_source_inputs_but_not_review_checkbox(self):
        p=self.ready()
        for change in (dict(has_blockout=False),dict(target_path=""),dict(source_ok=False)):
            self.assertTrue(self.check(p,**change))
        p.replica_reviewed=False;self.assertEqual(self.check(p),"")
    def test_changed_same_path_file_is_rejected(self):
        p=self.ready();self.target.write_bytes(b"replacement")
        self.assertIn("不一致",self.check(p))
    def test_ready_replica_passes(self):
        self.assertEqual(self.check(self.ready()),"")
    def test_input_lists_do_not_cross_modes(self):
        style=NS(source_path="environment.png",instruction="warm",image=None)
        p=props(style_images=[style],sb_image_path=str(self.target))
        self.assertEqual(OPS["_reference_paths"](p)["style"],["environment.png"])
        self.assertEqual(OPS["_sb_reference"](p),("",""))
        p.workflow_mode=WF.REFERENCE
        self.assertEqual(OPS["_reference_paths"](p)["style"],[str(self.target)])
        self.assertEqual(len(OPS["_reference_instructions"](p)["style"]),1)
        p.sb_image_path=""
        self.assertEqual(OPS["_reference_paths"](p)["style"],[])
    def test_environment_never_reads_blockout_source_or_band_notes(self):
        p=props();c=context(p)
        self.assertEqual(OPS["_sb_generation_reference"](c,p),("",""))
        self.assertEqual(OPS["_sb_ground_band_line"](c,p),"")
    def test_reference_source_hash_is_verified(self):
        p=self.ready();c=context(p);obj=Item("shell")
        block=Collection("Owned",[obj],tags={"wondful_scene_blockout":1,
            "wondful_sb_reference_image":str(self.target),"wondful_sb_reference_sha256":WF.file_hash(self.target)})
        c.scene.collection.children_recursive.append(block);p.massing_collection=block
        self.assertEqual(OPS["_sb_generation_reference"](c,p)[0],str(self.target))
        self.target.write_bytes(b"changed")
        self.assertEqual(OPS["_sb_generation_reference"](c,p)[0],"")
    def test_prepared_snapshot_may_differ_from_valid_original_target(self):
        p=self.ready();c=context(p);obj=Item("shell")
        snapshot=Path(self.tmp.name)/"prepared.png";snapshot.write_bytes(b"resized-and-reencoded-png")
        block=Collection("Owned",[obj],tags={"wondful_scene_blockout":1,
            "wondful_sb_reference_image":str(snapshot),"wondful_sb_reference_sha256":WF.file_hash(snapshot),
            "wondful_sb_original_sha256":WF.file_hash(self.target)})
        c.scene.collection.children_recursive.append(block);p.massing_collection=block
        self.assertEqual(OPS["_sb_generation_reference"](c,p)[0],str(snapshot))
        self.assertEqual(OPS["workflow_preflight"](c,p),"")
        self.target.write_bytes(b"replaced-original")
        self.assertIn("不一致",OPS["workflow_preflight"](c,p))
    def test_environment_generation_never_uses_replica_view_list(self):
        p=props(workflow_multiview=True);c=context(p)
        other=Item("Other",kind="CAMERA");c.scene.objects.append(other);p.mv_cameras=[NS(camera=other,enabled=True)]
        with patch.object(MV,"scene_blockout_exists",return_value=True):
            self.assertEqual(OPS["generate_views"](c,p),[])
            op=OPS["WONDFUL_OT_generate"]();original=c.scene.camera
            self.assertTrue(op._plan_views(c,p,c.preferences.addons[PACKAGE].preferences))
            self.assertIs(original,c.scene.camera)
    def test_reference_multiview_requires_explicit_toggle(self):
        p=self.ready();c=context(p)
        p.mv_cameras=[NS(camera=c.scene.camera,enabled=True)]
        with patch.object(MV,"scene_blockout_exists",return_value=True):
            self.assertEqual(OPS["generate_views"](c,p),[])
            p.workflow_multiview=True
            self.assertEqual(OPS["generate_views"](c,p),[c.scene.camera])
    def test_generate_dispatches_mandatory_polish_without_camera_change(self):
        p=props(prompt="manual",style_prompt_dirty=True);c=context(p);calls=[];original=c.scene.camera
        bpy.ops=NS(wondful=NS(ai_render=lambda *a: calls.append("render") or {"FINISHED"},
                              polish_prompt=lambda *a: calls.append("polish") or {"FINISHED"}))
        op=OPS["WONDFUL_OT_generate"]()
        self.assertEqual(op.execute(c),{"FINISHED"});self.assertEqual(calls,["polish"])
        self.assertIs(original,c.scene.camera);self.assertEqual(p.prompt,"manual")
    def test_replica_rejection_happens_before_any_provider_call(self):
        p=props(workflow_mode=WF.REFERENCE);c=context(p)
        op=OPS["WONDFUL_OT_ai_render"]()
        with patch.dict(OPS,{"_analysis_provider":lambda *a: self.fail("provider called before local guard")}):
            self.assertEqual(op.execute(c),{"CANCELLED"})


class IsolationTests(unittest.TestCase):
    def make(self):
        c=context();owned=Item("shell",{"wondful_sb_group":1});normal=Item("Wondful manually named")
        inferred=Item("inferred",{"wondful_detail_inferred":1});already=Item("hidden",{"wondful_sb_wrap":1})
        already.hide_render=True;already.hidden=True
        product=c.scene.wondful_ai.product_collection.all_objects[0];product.tags["wondful_sb_group"]=1
        c.scene.objects.extend([owned,normal,inferred,already])
        return c,owned,normal,inferred,already,product
    def test_only_owned_geometry_hidden_and_restored(self):
        c,o,n,i,a,p=self.make()
        with WF.isolate_environment_capture(c):
            self.assertTrue(o.hidden and o.hide_render);self.assertTrue(i.hide_render)
            self.assertFalse(n.hide_render);self.assertFalse(p.hide_render)
        self.assertFalse(o.hidden or o.hide_render);self.assertFalse(i.hide_render)
        self.assertTrue(a.hidden and a.hide_render)
    def test_exception_restores_visibility(self):
        c,o,*_=self.make()
        with self.assertRaises(RuntimeError):
            with WF.isolate_environment_capture(c):
                raise RuntimeError("render failed")
        self.assertFalse(o.hidden or o.hide_render)
    def test_nested_capture_preserves_outer_state(self):
        c,o,*_=self.make()
        with WF.isolate_environment_capture(c):
            with WF.isolate_environment_capture(c):self.assertTrue(o.hide_render)
            self.assertTrue(o.hide_render)
        self.assertFalse(o.hide_render)
    def test_reference_capture_does_not_hide_geometry(self):
        c,o,*_=self.make();c.scene.wondful_ai.workflow_mode=WF.REFERENCE
        with WF.isolate_environment_capture(c):self.assertFalse(o.hide_render)
    def test_tagged_collection_descendants_are_isolated(self):
        c=context();o=Item("untagged member");sub=Collection("sub",[o])
        coll=Collection("owned",children=[sub],tags={"wondful_scene_blockout":1})
        c.scene.collection.children_recursive.extend([coll,sub]);c.scene.objects.append(o)
        with WF.isolate_environment_capture(c):self.assertTrue(o.hide_render)
        self.assertFalse(o.hide_render)
    def test_decorator_is_on_all_submission_capture_entrypoints(self):
        for file,names in (("viewport_capture.py",["capture_camera_reference","render_clay_base","render_detail_pass","render_product_mask"]),
                           ("structure_packet.py",["build_structure_packet"])):
            tree=ast.parse((PKG/file).read_text(encoding="utf-8"))
            for name in names:
                fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
                self.assertIn("isolated_capture",[ast.unparse(n) for n in fn.decorator_list])


class ComparisonTests(unittest.TestCase):
    def test_target_blockout_available_before_ai_result(self):
        p=props(workflow_mode=WF.REFERENCE,compare_pair="TARGET_BLOCKOUT",sb_reference_image=object(),sb_preview_image=object())
        self.assertEqual(WF.compare_images(p),(p.sb_reference_image,p.sb_preview_image))
        self.assertIsNone(p.result_image)
    def test_environment_ignores_old_target_compare_selection(self):
        p=props(compare_pair="TARGET_RESULT",viewport_image=object(),result_image=object(),sb_reference_image=object())
        self.assertEqual(WF.compare_images(p),(p.viewport_image,p.result_image))
    def test_target_final_pair(self):
        p=props(workflow_mode=WF.REFERENCE,compare_pair="TARGET_RESULT",sb_reference_image=object(),result_image=object())
        self.assertEqual(WF.compare_images(p),(p.sb_reference_image,p.result_image))


class FakeImage:
    def __init__(self, name="Image", width=160, height=90):
        self.name=name;self.size=(width,height);self.filepath="/tmp/"+name+".png"
    def preview_ensure(self):
        pass


class Layout:
    """API-call / RNA-name audit only. This is not a screen rendering."""
    def __init__(self, log=None, parent=None):
        self.log=[] if log is None else log;self.parent=parent;self.enabled=True
    def effective_enabled(self):
        return self.enabled and (self.parent.effective_enabled() if self.parent else True)
    def child(self, *a, **kw):return Layout(self.log,self)
    row=column=box=split=grid_flow=child
    def label(self,*a,**kw):self.log.append(("label",kw.get("text",""),self.effective_enabled()))
    def prop(self,obj,attr,**kw):
        if not hasattr(obj,attr):raise AssertionError("Unknown RNA field: "+attr)
        self.log.append(("prop",attr,self.effective_enabled()))
    def prop_search(self,obj,attr,*a,**kw):self.prop(obj,attr,**kw)
    def operator(self,idname,**kw):
        self.log.append(("operator",idname,self.effective_enabled(),kw.get("text","")))
        return NS()
    def separator(self,*a,**kw):pass
    separator_spacer=separator
    def template_icon(self,*a,**kw):pass
    template_ID_preview=template_icon
    template_ID=template_icon
    template_list=template_icon
    def progress(self,*a,**kw):pass
    def context_pointer_set(self,*a,**kw):pass


def ui_module():
    stub("properties",MAX_REFERENCES_PER_KIND=8)
    stub("prompt_profiles",needs_target_refresh=lambda *a:False)
    stub("prompt_engine",needs_reference_policy_refresh=lambda p:False)
    stub("viewport_capture",camera_output_dimensions=lambda context:(1600,900))
    stub("operators",generate_needs_polish=OPS["generate_needs_polish"],
         _reference_paths=OPS["_reference_paths"],provider_variant_cap=lambda p:2 if p=="antigravity" else 4,
         render_variant_count=lambda p,props:min(int(props.render_count),2 if p=="antigravity" else 4),
         views_plan=lambda p:{},keep_exact_state=lambda c,p:([],0))
    stub("update_ops",banner_manifest=lambda:None,STATE={"phase":"IDLE","message":""},
         local_version=lambda:"1.2.4",busy=lambda:False)
    stub("precise_setup",cached_status=lambda:{"ready":True},APPROX_MB=0)
    u=load("ui");u._supports_placeholder=lambda:False
    return u


class UILayoutContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.ui=ui_module()
    def draw(self,p):
        c=context(p);layout=Layout();panel=NS(layout=layout)
        self.ui.WONDFUL_PT_main._draw_main(panel,c)
        return layout.log
    def test_main_mode_controls_are_separate(self):
        env=self.draw(props())
        ref=self.draw(props(workflow_mode=WF.REFERENCE))
        self.assertFalse(any(x[0]=="prop" and x[1]=="sb_image_path" for x in env))
        self.assertFalse(any(x[0]=="operator" and x[1]=="wondful.scene_blockout_build" for x in env))
        self.assertTrue(any(x[:2]==("operator","wondful.target_load") for x in ref))
        self.assertTrue(any(x[0]=="operator" and x[1]=="wondful.scene_blockout_build" for x in ref))
        for calls in (env, ref):
            self.assertEqual(sum(x[:2] == ("operator", "wondful.set_workflow_mode") for x in calls), 2)
        for calls in (env,ref):
            for prop in ("product_collection","camera","output_directory","long_edge","render_count"):
                self.assertTrue(any(x[0]=="prop" and x[1]==prop for x in calls),prop)
            self.assertTrue(any(x[0]=="operator" and x[1]=="wondful.polish_prompt" for x in calls))
    def test_replica_render_no_longer_requires_review_in_ui(self):
        p=props(workflow_mode=WF.REFERENCE,codex_account_state="LOGGED_IN",sb_image_path="target.png",sb_built_target_hash="built-target")
        with patch.object(MV,"scene_blockout_exists",return_value=True):
            before=self.draw(p);p.replica_reviewed=True;after=self.draw(p)
        self.assertTrue(next(x[2] for x in before if x[:2]==("operator","wondful.generate")))
        self.assertTrue(next(x[2] for x in after if x[:2]==("operator","wondful.generate")))
    def test_ui_draw_scenario_matrix(self):
        scenarios=0
        for mode in (WF.ENVIRONMENT,WF.REFERENCE):
            for provider in ("CODEX","ANTIGRAVITY"):
                for tab in ("MODEL","OUTPUT","FINISH","PRODUCT","SCENE","MULTIVIEW","DIAG"):
                    for state in ("IDLE","RENDERING"):
                        with self.subTest(mode=mode,provider=provider,tab=tab,state=state):
                            p=props(workflow_mode=mode,analysis_provider=provider,ui_show_advanced=True,ui_adv_tab=tab,
                                    codex_account_state="LOGGED_IN",antigravity_account_state="LOGGED_IN",status=state)
                            self.draw(p);scenarios+=1
        self.assertEqual(scenarios,56)
    def test_preview_is_available_before_final_result(self):
        p=props(workflow_mode=WF.REFERENCE,sb_reference_image=FakeImage("Target"),sb_preview_image=FakeImage("Blockout"))
        calls=self.draw(p)
        self.assertTrue(any(x[:2]==("operator","wondful.open_compare_workspace") for x in calls))
        self.assertIsNone(p.result_image)
    def test_result_label_records_actual_workflow(self):
        p=props(result_image=FakeImage("Result"),result_workflow=WF.REFERENCE)
        calls=self.draw(p)
        self.assertTrue(any(x[0]=="label" and "参考图复刻" in x[1] and "结果" in x[1] for x in calls))


class AdditionalBoundaryTests(unittest.TestCase):
    def test_environment_revision_cannot_invalidate_replica_draft(self):
        p=props(workflow_mode=WF.REFERENCE,style_reference_revision=77)
        self.assertEqual(WF.reference_revision(p),0)
        p.workflow_mode=WF.ENVIRONMENT
        self.assertEqual(WF.reference_revision(p),77)

    def test_new_brief_does_not_reuse_older_ai_prose(self):
        p=props(prompt="old sunny scene",last_ai_prompt="old sunny scene",
                scene_brief="snowy night",scene_brief_used="sunny day")
        self.assertEqual(WF.render_text(p,False),"snowy night")

    def test_current_ai_prose_remains_usable(self):
        p=props(prompt="expanded scene",last_ai_prompt="expanded scene",
                scene_brief="same brief",scene_brief_used="same brief")
        self.assertEqual(WF.render_text(p,False),"expanded scene")

    def test_excluded_view_layer_object_restores_render_visibility(self):
        class Excluded(Item):
            def hide_get(self,**kw):raise RuntimeError("No object base in layer")
        c=context();obj=Excluded("Hidden layer shell",{"wondful_sb_group":1})
        c.scene.objects.append(obj)
        with WF.isolate_environment_capture(c):self.assertTrue(obj.hide_render)
        self.assertFalse(obj.hide_render)

    def test_owned_camera_and_light_never_hidden(self):
        c=context()
        objects=[Item("Owned "+kind,{"wondful_sb_group":1},kind) for kind in ("CAMERA","LIGHT")]
        c.scene.objects.extend(objects)
        with WF.isolate_environment_capture(c):
            self.assertTrue(all(not obj.hide_render and not obj.hidden for obj in objects))

    def test_render_preflight_validates_actual_blockout_origin(self):
        with tempfile.TemporaryDirectory() as tmp:
            target=Path(tmp)/"target.png";target.write_bytes(b"target")
            digest=WF.file_hash(target)
            p=props(workflow_mode=WF.REFERENCE,sb_image_path=str(target),sb_built_target_hash=digest,replica_reviewed=True)
            c=context(p);obj=Item("shell")
            block=Collection("Owned",[obj],tags={"wondful_scene_blockout":1,
                "wondful_sb_reference_image":str(target),"wondful_sb_reference_sha256":digest,
                "wondful_sb_original_sha256":digest})
            p.massing_collection=block;c.scene.collection.children_recursive.append(block)
            c.scene.objects.append(obj)
            self.assertEqual(OPS["workflow_preflight"](c,p),"")
            block.tags["wondful_sb_original_sha256"]="from-another-target"
            self.assertTrue(OPS["workflow_preflight"](c,p))

    def test_legacy_blockout_requires_explicit_rebuild_not_silent_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            target=Path(tmp)/"target.png";target.write_bytes(b"target")
            p=props(workflow_mode=WF.REFERENCE,sb_image_path=str(target),sb_built_target_hash=WF.file_hash(target),replica_reviewed=True)
            c=context(p);block=Collection("Legacy",[Item("shell")],tags={"wondful_scene_blockout":1,
                "wondful_sb_reference_image":str(target),"wondful_sb_reference_sha256":WF.file_hash(target)})
            p.massing_collection=block;c.scene.collection.children_recursive.append(block)
            self.assertTrue(OPS["workflow_preflight"](c,p))

    def test_review_helper_uses_current_camera_and_real_product(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=props(workflow_mode=WF.REFERENCE);c=context(p)
            camera=c.scene.camera
            camera.matrix_world=[[1.,0.,0.,3.],[0.,1.,0.,4.],[0.,0.,1.,5.],[0.,0.,0.,1.]]
            camera.data.shift_x=0.13;camera.data.shift_y=-0.07
            block=Collection("Owned",[Item("shell")],tags={"wondful_scene_blockout":1})
            p.massing_collection=block
            calls=[];packed=[]
            helper=function_namespace("_render_replica_preview")
            helper.update(camera_output_dimensions=lambda c:(1600,900),
                load_image=lambda *a:NS(pack=lambda:packed.append(True)))
            sbb=types.ModuleType(PACKAGE+".scene_blockout_build")
            sbb.render_preview=lambda *a,**kw:calls.append((a,kw)) or True
            with patch.dict(sys.modules,{sbb.__name__:sbb}), patch.object(package,"scene_blockout_build",sbb,create=True), \
                 patch.object(WF,"target_changed",lambda *a:None):
                result=helper["_render_replica_preview"](c,p,Path(tmp))
            self.assertTrue(result.endswith("review_blockout.png"));self.assertEqual(packed,[True])
            self.assertIs(calls[0][1]["product_collection"],p.product_collection)
            info=calls[0][0][1]
            self.assertEqual(info["shift_x"],0.13);self.assertEqual(info["shift_y"],-0.07)
            self.assertEqual(info["cam_world"],camera.matrix_world)
            self.assertIs(c.scene.camera,camera);self.assertEqual(p.compare_pair,"TARGET_BLOCKOUT")

    def finish_fixture(self, fail_preview=False):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
        p=props(workflow_mode=WF.REFERENCE,sb_complete=False,replica_reviewed=True,sb_preview_image=object())
        c=context(p);ns=function_namespace(classes=("WONDFUL_OT_scene_blockout_build",))
        calls=[]
        def preview(*a):
            calls.append(a)
            if fail_preview:raise RuntimeError("preview renderer unavailable")
            return str(Path(tmp.name)/"preview.png")
        ns.update(_render_replica_preview=preview,_status=lambda p,s:setattr(p,"status",s),_redraw_all=lambda:None)
        op=ns["WONDFUL_OT_scene_blockout_build"]();op._origin_scene=c.scene
        op._sb={"session":Path(tmp.name),"scores":[],"cached":False,"t0":__import__("time").time(),
                "target_sha256":"original-target","correct":False}
        sb=types.ModuleType(PACKAGE+".scene_blockout");sb.summary=lambda info:"Blockout complete"
        db=types.ModuleType(PACKAGE+".detail_blockout");db.remove_inferred=lambda:None
        with patch.dict(sys.modules,{sb.__name__:sb,db.__name__:db}), \
             patch.object(package,"scene_blockout",sb,create=True),patch.object(package,"detail_blockout",db,create=True):
            outcome=op._finish(c,{"kind":"outdoor"},1)
        metadata=json.loads((Path(tmp.name)/"scene_blockout_info.json").read_text())
        return p,outcome,calls,metadata

    def test_build_without_ai_correction_still_creates_review_preview(self):
        p,outcome,calls,metadata=self.finish_fixture()
        self.assertEqual(outcome,{"FINISHED"});self.assertEqual(len(calls),1)
        self.assertFalse(p.replica_reviewed);self.assertEqual(p.sb_built_target_hash,"original-target")
        self.assertEqual(p.status,"IDLE");self.assertTrue(metadata["review_preview"]["ok"])

    def test_review_render_failure_is_reported_not_marked_approved(self):
        p,outcome,calls,metadata=self.finish_fixture(fail_preview=True)
        self.assertEqual(outcome,{"FINISHED"});self.assertFalse(p.replica_reviewed)
        self.assertIsNone(p.sb_preview_image);self.assertIn("预览未完成",p.last_sb_note)
        self.assertFalse(metadata["review_preview"]["ok"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
