"""Pure contracts for source material-plan catalog and strict response parser."""
import json
import math
import unittest
from types import SimpleNamespace as NS

from test_316_regressions import load_package_module

MP = load_package_module("reference_material_plan")


class Thing(dict):
    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)


def material(name, *, linked=False, linked_values=(), library=None, keep=None, values=None):
    values = values or {"Roughness": .42, "Metallic": .1, "Base Color": (.2, .3, .4, 1)}
    sockets = [NS(name=key, is_linked=(linked or key in linked_values), default_value=value) for key, value in values.items()]
    node = NS(type="BSDF_PRINCIPLED", inputs=sockets)
    value = Thing(name=name, use_nodes=True, node_tree=NS(nodes=[node]), library=library)
    if keep is not None:
        value["wondful_keep_exact"] = keep
    return value


class MaterialPlanTests(unittest.TestCase):
    def setUp(self):
        self.clean = material("wheel trim")
        self.paint = material("body paint")
        self.linked = material("textured logo", linked=True)
        self.keep = material("badge", keep=True)
        obj = Thing(name="product", type="MESH", hide_render=False,
                    material_slots=[NS(material=value) for value in (self.clean, self.paint, self.linked, self.keep)])
        self.context = NS()
        self.props = NS(product_collection=NS(all_objects=[obj]))
        self.catalog = MP.catalog(self.context, self.props)

    def reply(self, entries):
        return json.dumps({"prompt": "保留结构，仅调整表面。", "materials": entries})

    def test_catalog_marks_paint_keep_and_linked_socket_protection_without_product_hardcoding(self):
        rows = {row["name"]: row for row in self.catalog}
        self.assertTrue(rows["body paint"]["paint"])
        self.assertTrue(rows["badge"]["keep"])
        self.assertFalse(rows["textured logo"]["linked"])
        self.assertIn("roughness", rows["textured logo"]["linked_values"])
        self.assertEqual(rows["textured logo"]["editable_values"], [])
        self.assertFalse(rows["wheel trim"]["paint"])
        self.assertEqual(rows["wheel trim"]["current_values"]["roughness"], .42)
        self.assertIn("CATALOG:", MP.polish_contract(self.catalog))

    def test_valid_plan_and_explicit_empty_plan(self):
        prompt, plan = MP.parse_reply(self.reply([]), self.catalog)
        self.assertEqual(prompt, "保留结构，仅调整表面。")
        self.assertEqual(plan, [])
        _, plan = MP.parse_reply(self.reply([{"name": "wheel trim", "values": {
            "roughness": .25, "metallic": 1, "base_color": [.1, .2, .3]}}]), self.catalog)
        self.assertEqual(plan[0]["values"]["metallic"], 1.0)

    def test_catalog_current_values_identify_a_nonempty_fixture_plan_without_authorizing_recolor(self):
        values = self.catalog[0]["current_values"]
        self.assertIn("base_color", values)
        prompt, plan = MP.parse_reply(self.reply([{"name": "wheel trim", "values": {"roughness": .21}}]), self.catalog)
        self.assertTrue(prompt)
        self.assertEqual(plan, [{"name": "wheel trim", "values": {"roughness": .21}}])

    def test_paint_base_color_is_immutable_but_its_other_pbr_values_are_allowed(self):
        _, plan = MP.parse_reply(self.reply([{"name": "body paint", "values": {"roughness": .18}}]), self.catalog)
        self.assertEqual(plan[0]["values"], {"roughness": .18})
        with self.assertRaises(ValueError):
            MP.parse_reply(self.reply([{"name": "body paint", "values": {"base_color": [.1, .2, .3]}}]), self.catalog)

    def test_base_color_texture_keeps_its_link_but_allows_unlinked_pbr(self):
        paint = material("body paint", linked_values=("Base Color",))
        obj = Thing(name="product", type="MESH", hide_render=False, material_slots=[NS(material=paint)])
        catalog = MP.catalog(NS(), NS(product_collection=NS(all_objects=[obj])))
        row = catalog[0]
        self.assertEqual(row["linked_values"], ["base_color"])
        self.assertIn("roughness", row["editable_values"])
        _, plan = MP.parse_reply(self.reply([{"name": "body paint", "values": {"roughness": .16}}]), catalog)
        self.assertEqual(plan[0]["values"], {"roughness": .16})
        with self.assertRaises(ValueError):
            MP.parse_reply(self.reply([{"name": "body paint", "values": {"base_color": [.1, .2, .3]}}]), catalog)

    def test_library_material_remains_wholly_protected(self):
        external = material("licensed trim", library=object())
        obj = Thing(name="product", type="MESH", hide_render=False, material_slots=[NS(material=external)])
        catalog = MP.catalog(NS(), NS(product_collection=NS(all_objects=[obj])))
        self.assertTrue(catalog[0]["linked"])
        self.assertEqual(catalog[0]["editable_values"], [])
        with self.assertRaises(ValueError):
            MP.parse_reply(self.reply([{"name": "licensed trim", "values": {"roughness": .2}}]), catalog)

    def test_unknown_paint_fallback_locks_only_every_base_color(self):
        plain = material("wheel trim")
        obj = Thing(name="product", type="MESH", hide_render=False, material_slots=[NS(material=plain)])
        catalog = MP.catalog(NS(), NS(product_collection=NS(all_objects=[obj])))
        self.assertTrue(catalog[0]["base_color_locked"])
        _, plan = MP.parse_reply(self.reply([{"name": "wheel trim", "values": {"metallic": .7}}]), catalog)
        self.assertEqual(plan[0]["values"]["metallic"], .7)
        with self.assertRaises(ValueError):
            MP.parse_reply(self.reply([{"name": "wheel trim", "values": {"base_color": [.1, .2, .3]}}]), catalog)

    def test_catalog_includes_visible_curve_and_excludes_hidden_render_geometry(self):
        curve = Thing(name="logo", type="FONT", hide_render=False, material_slots=[NS(material=material("badge metal"))])
        hidden = Thing(name="hidden", type="MESH", hide_render=True, material_slots=[NS(material=material("hidden"))])
        catalog = MP.catalog(NS(), NS(product_collection=NS(all_objects=[curve, hidden])))
        self.assertEqual([row["name"] for row in catalog], ["badge metal"])

    def test_malformed_nonfinite_unknown_duplicate_and_protected_reject(self):
        cases = [
            "not json",
            json.dumps({"prompt": "x", "materials": [{"name": "wheel trim", "values": {"roughness": math.nan}}]}),
            self.reply([{"name": "missing", "values": {"roughness": .2}}]),
            self.reply([{ "name": "wheel trim", "values": {"roughness": .2}}, {"name": "wheel trim", "values": {"metallic": .2}}]),
            self.reply([{ "name": "body paint", "values": {"base_color": [.1, .2, .3]}}]),
            self.reply([{ "name": "textured logo", "values": {"roughness": .2}}]),
            self.reply([{ "name": "badge", "values": {"roughness": .2}}]),
        ]
        for reply in cases:
            with self.subTest(reply=reply), self.assertRaises(ValueError):
                MP.parse_reply(reply, self.catalog)


if __name__ == "__main__":
    unittest.main()
