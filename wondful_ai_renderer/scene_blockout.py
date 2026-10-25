"""1.2.3 ④ 场景粗模（可选）· 从参考图生成场景粗模 — analysis side (no bpy).

One environment photo → strict-JSON scene layout (camera, heightfield terrain, roads, cliffs,
distant bands, foliage, primitives, product spot, palette) from the reasoning model with the
user's Codex login.  ``scene_blockout_build`` turns the JSON into Blender blocks.

The source photo is analysed here and recorded on the resulting blockout. When that
specific blockout participates in generation, its source may accompany it as a
continuous-surface/material/lighting reference. The calibrated Blender camera and
the user's product geometry remain authoritative; ordinary environment references
retain their existing appearance-only role.
Ported from the accepted v4 prototype (env_blockout/analyze_v3.py + analyze_v4.py) with:
- any frame aspect (the prototype assumed 16:9);
- the real product size in the product-spot rule;
- Chinese short names (``label_zh``) per element and a Chinese season / scene line, so the
  generation prompt can name every block;
- an optional correction round fed by a numeric fit score (sky IoU + edge F-score, numpy only),
  kept only when the score improves.
"""
from __future__ import annotations

import copy
import json
import math
import re

import numpy as np

COLLECTION_NAME = "Wondful 场景粗模"
CAMERA_NAME = "Wondful 参考相机"
TEXT_NAME = "wondful_scene_blockout.json"
SCENE_NOTE_PROP = "wondful_sb_scene_note"
GROUP_PROP = "wondful_sb_group"
SCHEMA_NAME = "wondful_scene_blockout_v1"
EFFORT_FIRST = "high"
EFFORT_CORRECT = "medium"
WORK_LONG = 1600          # analysis image long edge
SCORE_W = 320             # fit-score working width
MIN_GAIN = 0.005          # the correction round is kept only when the fit score rises by more than this

# ---------------------------------------------------------------------------------------- schema

PRIM_TYPES = ["building", "tree", "bush", "hedge", "wall", "fence", "pole", "vehicle", "rock", "water_plane", "awning",
              "person", "sign", "ceiling", "furniture", "window", "rug", "lamp", "plant", "curtain"]
PRIM_SHAPES = ["box", "gable", "cylinder", "cone", "sphere", "dome"]
_num = {"type": "number"}
_s = {"type": "string"}
_pt2 = {"type": "object", "additionalProperties": False, "required": ["x", "y"], "properties": {"x": _num, "y": _num}}
_pt3 = {"type": "object", "additionalProperties": False, "required": ["x", "y", "z"],
        "properties": {"x": _num, "y": _num, "z": _num}}
_spt = {"type": "object", "additionalProperties": False, "required": ["x_frac", "y_frac"],
        "properties": {"x_frac": _num, "y_frac": _num}}

SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["scene_kind", "scene_summary", "environment_description", "environment_zh", "palette", "camera",
                 "terrain", "roads", "mountain_bands", "foliage_masses", "primitives", "product_spot", "cliffs",
                 "surface_zones", "reference_subject", "ribbons"],
    "properties": {
        "scene_kind": {"type": "string", "enum": ["outdoor", "interior"]},
        "scene_summary": _s,
        "environment_description": _s,
        "environment_zh": _s,
        "palette": {"type": "object", "additionalProperties": False,
                    "required": ["season", "season_evidence", "grass_hex", "grass_words", "foliage_hex", "foliage_words",
                                 "ground_hex", "ground_words", "sky_words", "snow_cover", "autumn_colours", "lock_zh"],
                    "properties": {
                        "season": {"type": "string", "enum": ["spring", "early_summer", "summer", "late_summer", "autumn",
                                                              "winter", "dry_season", "wet_season", "not_applicable"]},
                        "season_evidence": _s, "grass_hex": _s, "grass_words": _s, "foliage_hex": _s,
                        "foliage_words": _s, "ground_hex": _s, "ground_words": _s, "sky_words": _s,
                        "snow_cover": {"type": "boolean"}, "autumn_colours": {"type": "boolean"}, "lock_zh": _s}},
        "camera": {"type": "object", "additionalProperties": False,
                   "required": ["height_m", "pitch_deg", "roll_deg", "hfov_deg", "horizon_y_frac"],
                   "properties": {"height_m": _num, "pitch_deg": _num, "roll_deg": _num, "hfov_deg": _num,
                                  "horizon_y_frac": _num}},
        "terrain": {"type": "object", "additionalProperties": False,
                    "required": ["far_m", "base_slope_deg", "base_slope_dir_deg", "smoothness", "surface", "label_zh",
                                 "bumps", "ridges", "has_sea", "sea_level_m"],
                    "properties": {
                        "far_m": _num, "base_slope_deg": _num, "base_slope_dir_deg": _num, "smoothness": _num,
                        "surface": _s, "label_zh": _s, "has_sea": {"type": "boolean"}, "sea_level_m": _num,
                        "bumps": {"type": "array", "maxItems": 16, "items": {
                            "type": "object", "additionalProperties": False,
                            "required": ["x", "y", "height_m", "radius_m"],
                            "properties": {"x": _num, "y": _num, "height_m": _num, "radius_m": _num}}},
                        "ridges": {"type": "array", "maxItems": 8, "items": {
                            "type": "object", "additionalProperties": False,
                            "required": ["points", "height_m", "width_m"],
                            "properties": {"points": {"type": "array", "minItems": 2, "maxItems": 8, "items": _pt2},
                                           "height_m": _num, "width_m": _num}}}}},
        "roads": {"type": "array", "maxItems": 5, "items": {
            "type": "object", "additionalProperties": False,
            "required": ["id", "kind", "label_zh", "width_m", "follow_terrain", "points", "surface"],
            "properties": {"id": _s, "kind": {"type": "string", "enum": ["road", "path", "sidewalk", "river", "plaza"]},
                           "label_zh": _s, "width_m": _num, "follow_terrain": {"type": "boolean"},
                           "points": {"type": "array", "minItems": 2, "maxItems": 12, "items": _pt3}, "surface": _s}}},
        "mountain_bands": {"type": "array", "maxItems": 3, "items": {
            "type": "object", "additionalProperties": False,
            "required": ["id", "kind", "label_zh", "distance_m", "depth_m", "profile", "surface"],
            "properties": {"id": _s, "kind": {"type": "string", "enum": ["rocky", "snowy", "forested", "grassy", "dune", "sea"]},
                           "label_zh": _s, "distance_m": _num, "depth_m": _num,
                           "profile": {"type": "array", "minItems": 3, "maxItems": 40, "items": _spt}, "surface": _s}}},
        "foliage_masses": {"type": "array", "maxItems": 12, "items": {
            "type": "object", "additionalProperties": False,
            "required": ["id", "kind", "label_zh", "x", "y", "crown_w", "crown_d", "crown_h", "crown_base_h", "trunk_w",
                         "lean_deg", "lumpiness", "outline", "material"],
            "properties": {"id": _s, "kind": {"type": "string", "enum": ["deciduous", "conifer", "palm", "shrub", "hedge",
                                                                          "forest_edge"]},
                           "label_zh": _s, "x": _num, "y": _num, "crown_w": _num, "crown_d": _num, "crown_h": _num,
                           "crown_base_h": _num, "trunk_w": _num, "lean_deg": _num, "lumpiness": _num,
                           "outline": {"type": "array", "maxItems": 40, "items": _spt}, "material": _s}}},
        "primitives": {"type": "array", "maxItems": 30, "items": {
            "type": "object", "additionalProperties": False,
            "required": ["id", "type", "shape", "label_zh", "x", "y", "z_offset", "w", "d", "h", "roof_h", "rot_z_deg",
                         "repeat", "repeat_dx", "repeat_dy", "material"],
            "properties": {
                "id": _s, "type": {"type": "string", "enum": PRIM_TYPES}, "shape": {"type": "string", "enum": PRIM_SHAPES},
                "label_zh": _s, "x": _num, "y": _num, "z_offset": _num, "w": _num, "d": _num, "h": _num, "roof_h": _num,
                "rot_z_deg": _num, "repeat": {"type": "integer"}, "repeat_dx": _num, "repeat_dy": _num, "material": _s}}},
        "product_spot": {"type": "object", "additionalProperties": False,
                         "required": ["x", "y", "rot_z_deg", "on", "reason"],
                         "properties": {"x": _num, "y": _num, "rot_z_deg": _num, "on": _s, "reason": _s}},
        "reference_subject": {"type": "object", "additionalProperties": False,
                              "required": ["present", "bbox", "ground_contact", "heading_deg", "heading_confidence"],
                              "properties": {
                                  "present": {"type": "boolean"},
                                  "bbox": {"type": "object", "additionalProperties": False,
                                           "required": ["x_min", "y_min", "x_max", "y_max"],
                                           "properties": {k: _num for k in ("x_min", "y_min", "x_max", "y_max")}},
                                  "ground_contact": _spt, "heading_deg": _num, "heading_confidence": _num}},
        "ribbons": {"type": "array", "maxItems": 12, "items": {
            "type": "object", "additionalProperties": False,
            "required": ["id", "label_zh", "sections", "thickness_m", "material"],
            "properties": {"id": _s, "label_zh": _s, "thickness_m": _num, "material": _s,
                           "sections": {"type": "array", "minItems": 2, "maxItems": 32, "items": {
                               "type": "object", "additionalProperties": False,
                               "required": ["left", "right"], "properties": {"left": _pt3, "right": _pt3}}}}}},
        "cliffs": {"type": "array", "maxItems": 4, "items": {
            "type": "object", "additionalProperties": False,
            "required": ["id", "label_zh", "base_points", "top_z_m", "side", "thickness_m", "lean_deg", "roughness",
                         "surface"],
            "properties": {"id": _s, "label_zh": _s, "base_points": {"type": "array", "minItems": 2, "maxItems": 12,
                                                                       "items": _pt3},
                           "top_z_m": _num, "side": {"type": "string", "enum": ["left", "right"]}, "thickness_m": _num,
                           "lean_deg": _num, "roughness": _num, "surface": _s}}},
        "surface_zones": {"type": "array", "maxItems": 6, "items": {
            "type": "object", "additionalProperties": False, "required": ["id", "kind", "points", "surface"],
            "properties": {"id": _s, "kind": {"type": "string", "enum": ["shrub_slope", "grass", "rock", "scree", "sand",
                                                                          "snow", "paving", "field"]},
                           "points": {"type": "array", "minItems": 3, "maxItems": 14, "items": _pt2}, "surface": _s}}},
    },
}

# ---------------------------------------------------------------------------------------- rules

RULES = """You are a layout reconstructor for a Blender add-on. From ONE photo, rebuild the scene as a coarse blockout whose
silhouettes line up with the photo when rendered from the same camera. Output strict JSON (schema given).

FRAME: the photo is {fw}x{fh} pixels (aspect {aspect_txt}). All screen fractions refer to this frame.
COORDINATES (metres, Z up): camera at x=0,y=0, z=height_m above the ground under it, looking along +Y (yaw 0).
x = right, y = forward (depth). pitch_deg negative = looking down. hfov_deg = horizontal FOV of the frame.
horizon_y_frac = true eye-level horizon in the frame (0 top, 1 bottom); it must agree with pitch:
horizon_y_frac = 0.5 + tan(pitch)/ (2*tan(vfov/2)) with tan(vfov/2)=tan(hfov/2)*{hw:.4f}.
Screen mapping of a point (x,y,z) for small pitch: x_frac ~= 0.5 + x/(2*y*tan(hfov/2)); y_frac ~= horizon_y_frac - (z-height_m)/(2*y*tan(vfov/2)).

SCENE_KIND: outdoor or interior (a room). For an interior: terrain = flat floor (all slopes 0, no bumps/ridges, far_m 60),
 no roads, no mountain_bands, no foliage_masses unless real indoor plants big enough; walls = box primitives type wall
 (thickness 0.2, full room height), ceiling = one box type ceiling covering the room at the room height (z_offset = room
 height, h 0.2), windows/glass doors = thin box type window placed in the wall plane (slightly in front of it), curtains =
 thin box type curtain, sofas/tables/TV units/cabinets/shelves = type furniture (box), rugs = very flat box type rug (h 0.02),
 lamps = type lamp, indoor plants = type plant. Typical room height 2.6-3.0 m, camera height ~1.2-1.6 m.

PALETTE (season lock; these words must also work when a source image cannot be attached): name the season you SEE (not what is typical),
 with evidence (leaf colour, snow, light, flowers). Give the colours as they appear in the photo with a hex estimate:
 grass_hex/grass_words (e.g. "#7FA34A fresh mid-green with a few dry yellow patches"), foliage_hex/foliage_words, ground_hex/
 ground_words (road, sand, snow, floor), sky_words. snow_cover = true only if snow is on the ground. autumn_colours = true only if
 the leaves in the photo are actually yellow/orange/red. Do not exaggerate dryness or autumn tones that are only hinted.
 lock_zh = ONE short Chinese sentence locking season, vegetation colour, ground colour, sky and light (e.g.
 "盛夏晴天正午：草地鲜绿、树叶浓绿，没有落叶和积雪，天空湛蓝少云"). For interiors: the light and colour scheme of the room.
 environment_description must use the same season/colour words; environment_zh = the same in Chinese, at most 80 characters.

LABEL_ZH (every road, band, foliage mass, primitive, cliff and the terrain): 2-8 Chinese characters naming what it really is
 in this photo, e.g. 红砖厂房, 玻璃幕墙写字楼, 沥青公路, 人行道, 棕榈树, 松树林, 沙丘, 远处雪山, 崖壁, 布艺沙发, 木茶几,
 落地窗, 白纱窗帘, 实木地板, 草地. Chinese only, no English, no numbers.

TERRAIN = one smooth heightfield (the builder makes a fine subdivided mesh; never use primitives for ground, hills, dunes, slopes):
 height(x,y) = tan(base_slope_deg) * (distance along base_slope_dir_deg; 0 = rises forward +Y, 90 = rises to +X)
             + sum over bumps: height_m * 0.5*(1+cos(pi*d/radius_m)) for d<radius_m (d = distance to (x,y)); negative = hollow
             + sum over ridges: height_m * same cosine falloff of distance to the ridge polyline, radius = width_m/2; negative = valley.
 smoothness 0..1 = extra relaxation. far_m = how far the terrain mesh reaches (street ~300, landscape 2000-5000); beyond it the
 ground continues flat at the far-edge height. The camera foot is at height(0,0). Sand dunes, grassy hills, meadows, embankments
 = bumps/ridges. surface = the real ground material(s) in words.
ROADS = ribbons: centreline points (x,y,z) from near to far (curves = several points, smooth spline), width_m.
 follow_terrain = true: the road is draped on the terrain (z ignored). follow_terrain = false: z is the road surface height
 relative to the camera foot (0) at each point, and the builder CARVES/fills the terrain to the road (cuttings, embankments,
 descending side streets, ramps, cliff roads). Use false whenever the road clearly climbs/descends differently from the land
 around it, or for a side branch going down/up. Model a junction / curve with 2 roads. kind sidewalk next to the road, kind plaza
 for paved squares (wide). surface = real material. A side or branch road is its own road entry with its own grade.
MOUNTAIN_BANDS = distant mountains / far ridgelines / far dune crests / far forest edges as a crest silhouette IN SCREEN SPACE:
 profile = crest points (x_frac 0..1 left->right, y_frac 0 top..1 bottom) traced along the actual skyline of that range
 (15-40 points); distance_m; depth_m. kind rocky/snowy = jagged, forested/grassy/dune = smoothed. kind sea = the sea surface
 seen to the horizon: profile = the visible sea horizon line (flat), distance_m = where the visible sea starts.
 Never use cones or pyramids for mountains.
FOLIAGE_MASSES = every tree crown / tree group / hedge / forest edge that covers a noticeable part of the frame (biggest first).
 Two modes:
 (a) outline (preferred for big, near, frame-edge or cropped trees and for continuous forest walls): outline = 8-40 screen points
     (x_frac,y_frac; may go below 0 or above 1 where the foliage leaves the frame) tracing the foliage silhouette as one closed
     polygon in order (include the trunk if visible); x,y = ground position of the mass centre (depth y sets its distance),
     crown_d = thickness. The builder extrudes exactly that silhouette at depth y, so the screen area matches the photo.
 (b) volume (outline = []): an irregular canopy of clustered lobes on a trunk: crown_w/crown_d/crown_h, crown_base_h = trunk
     height under the crown, trunk_w, lean_deg (+ leans to +x), lumpiness 0..1. Use for free-standing mid-distance trees.
 One foliage mass per tree or per clearly separate tree group (max 12). Overhanging canopies over a road = separate crowns on
 the left and on the right, each with its own outline; leave the visible gaps between crowns as gaps; never one merged polygon
 spanning several trees or both sides of a road. kind conifer in volume mode gives a tapered lumpy spire. Small distant
 conifers can still be primitives tree/cone.
RIBBONS (max 12): continuous curved walls, broad twisting concrete bands, overhead bridges and flared shells must use
 ribbons rather than stacks of cylinders or disconnected boxes. Each ribbon is ONE connected ruled surface with 2-32
 ordered cross-sections; each section has left/right 3D boundary points (metres, Z relative to the camera-foot ground).
 Trace both physical edges through the bend; keep left/right correspondence continuous, with extra sections around turns.
 The builder interpolates between paired edges and adds thickness_m. Different crossing bands are separate ribbons;
 preserve their gaps and front/back order. For a flared wall the two edges can run from ground to its upper rim. Describe
 only visible evidence; this is a view-matched approximation, not a unique recovery of hidden geometry. Use [] if absent.
 Do not also duplicate a ribbon as primitives. material describes its real surface; label_zh names the structure.
PRIMITIVES (max 30, biggest silhouettes first; they sit ON the terrain: z_offset = 0 normally):
 x,y = footprint centre; w along local X, d along local Y, h = height (for gable: eave height), roof_h = gable roof height
 (ridge along local X; 0 for other shapes), rot_z_deg = rotation about Z.
 shapes: box (building masses, walls, cars, furniture), gable (pitched-roof building), cylinder (trunks, poles, lamps), cone ONLY
 for small conifer trees, sphere (small bushes), dome (boulders). repeat>1 = a row of copies stepping by (repeat_dx, repeat_dy).
 Utility poles, posts, lamp posts = shape cylinder, type pole, w = d = 0.15-0.4 m, never boxes or slabs.
 material = what it really is (e.g. "two-storey red-brick industrial hall, arched green-framed windows").
 MOVING THINGS: people / pedestrians / crowds = type person; other cars, taxis, buses, bicycles, motorbikes = type
 vehicle; animals = type person with an animal label_zh. The builder never turns these into blocks (only the static scene
 is rebuilt), so list them only when they matter for the product spot, and never let them displace a static element.
CLIFFS: a rock face rising near-vertically (steeper than ~60 deg) beside a road, above a beach or over a valley is a cliff entry,
 NOT terrain bumps/ridges and NOT a mountain band. base_points = the foot of the face from near to far (x, y, z = height of the
 foot relative to the camera foot; for a road cut into a cliff = the inner road-edge height, placed exactly along the inner
 edge of that road), top_z_m = the top of the face relative to the camera foot, side = on which side of the base line (walking
 near -> far) the rock mass lies, thickness_m, lean_deg (0 vertical, + leans back away from the road, negative = overhang),
 roughness 0..1, surface. A road ledge cut into a cliff = a road with follow_terrain false whose inner edge runs along the cliff
 base; on the outer side the land continues as TERRAIN - never merge road, cliff and the slope below into one mass. [] if none.
SURFACE_ZONES: ground-plane polygons (x, y in metres, 3-14 points, in order) giving terrain areas their own material:
 shrub_slope, grass, rock, scree, sand, snow, paving, field. [] when the whole ground is one material.
TERRAIN has_sea / sea_level_m: when sea or a lake is visible, has_sea = true and sea_level_m = the water height relative to the
 camera foot (negative when the water is below). The builder floods every terrain point below that level with water, so the
 terrain MUST fall below sea_level_m where the photo shows water. No water_plane primitive when has_sea is true.
 has_sea false and sea_level_m 0 otherwise.
REFERENCE_SUBJECT: when the photo has a clear main product/hero car, present=true. Measure its tight bbox in normalized
 screen coordinates (x_min,y_min,x_max,y_max; 0 left/top, 1 right/bottom), and ground_contact = the projection of the
 centre of its ground footprint (for a car, midpoint between the wheel contact pairs, not the bottom of its bumper).
 heading_deg is the GROUND-PLANE long-axis rotation about +Z: 0 along +Y, +90 towards -X, -90 towards +X; it is NOT an
 image slope. The target's longitudinal axis is matched modulo 180 degrees; do not claim the front direction can be
 recovered from an undirected target bounding box. heading_confidence is 0..1. Keep the reference's view angle and
 placement even when the target is taller/wider than the photographed car; never reshape an SUV into a coupe.
 For present=false use a zero bbox, ground_contact (0.5,0.8), heading_deg 0 and heading_confidence 0.
PRODUCT_SPOT: if reference_subject.present, choose x,y and rot_z_deg to match those measured image anchors and heading.
 The builder will fit the real target's projected width and ground contact, without forcing its height to match.
 Only when there is NO reference subject, choose where the user's product would naturally stand, visible and unoccluded, on open ground the camera
 sees in the lower-middle part of the frame. The product is {product_words}, about {pw:.2f} m wide x {pd:.2f} m deep x
 {ph:.2f} m high. Choose x,y = its centre on the ground so it spans about 15-45 % of the frame width (never touching the frame
 edges). The 15-45% suggestion applies ONLY when reference_subject.present=false. Place it on a road / plaza / flat sand /
 open floor with nothing standing on that spot; rot_z_deg = heading of its long axis
 (0 = along +Y; along the road direction there), on = what surface, reason = one sentence.
ENVIRONMENT_DESCRIPTION: 3-6 sentences of real materials, architecture/vegetation species, region, season, weather, sky,
 light direction and colour temperature, so a photoreal image can be made from the blockout WITHOUT seeing the photo.
 Describe only the static scene: never mention people, pedestrians, crowds, traffic, other cars, bicycles or animals
 (environment_zh and lock_zh too).
Use real-world scale cues (lane 3.2 m, storey 3 m, car 4.5x1.8x1.5 m, conifer 10-30 m, door 2.1 m, sofa 0.85 m high).
Before answering, mentally project each element with the camera and check its outline lands where it is in the photo."""

CORRECTION = """CORRECTION ROUND. Below: your previous JSON, its clay render from that camera (same frame; white/transparent =
sky / background), and a 50% overlay of the render on the photo.
Fit score of the previous round (0..1, higher is better): {scores}
Measured differences (fractions of the frame; x 0 left..1 right, y 0 top..1 bottom):
{hints}
Fix in this order: camera (horizon, pitch, FOV) if the horizon/vanishing point is off; then the skyline (mountain band profiles,
building tops, foliage outlines) where sky/non-sky disagree; then large region extents; then details. Keep what already fits
(do not change an element whose region already matches). Keep the palette and the label_zh names unless they are wrong.
Return the full corrected JSON.
Previous JSON:
{prev}"""

# ---------------------------------------------------------------------------------------- labels

TYPE_ZH = {"building": "建筑", "tree": "树", "bush": "灌木", "hedge": "绿篱", "wall": "墙", "fence": "围栏",
           "pole": "杆子", "vehicle": "路边车辆", "rock": "岩石", "water_plane": "水面", "awning": "遮阳篷",
           "person": "行人", "sign": "指示牌", "ceiling": "天花板", "furniture": "家具", "window": "窗",
           "rug": "地毯", "lamp": "灯", "plant": "绿植", "curtain": "窗帘"}
INTERIOR_TYPE_ZH = {"wall": "墙面"}
ROAD_ZH = {"road": "道路", "path": "小路", "sidewalk": "人行道", "river": "河流", "plaza": "广场"}
BAND_ZH = {"rocky": "远山", "snowy": "远处雪山", "forested": "远处林带", "grassy": "远处草坡", "dune": "远处沙丘",
           "sea": "海面"}
FOLIAGE_ZH = {"deciduous": "树", "conifer": "松树", "palm": "棕榈树", "shrub": "灌木", "hedge": "绿篱",
              "forest_edge": "树林"}
SEASON_ZH = {"spring": "春季", "early_summer": "初夏", "summer": "夏季", "late_summer": "夏末", "autumn": "秋季",
             "winter": "冬季", "dry_season": "旱季", "wet_season": "雨季", "not_applicable": ""}
_CJK = re.compile(r"[\u3400-\u9fff]")


def clean_label(text, fallback="", limit=12) -> str:
    """Model label → a short Chinese name (CJK only; no digits / latin), else ``fallback``."""
    s = re.sub(r"[^\u3400-\u9fff（）、·]", "", str(text or ""))
    s = s.strip("（）、·")[:limit]
    return s if _CJK.search(s) else fallback


def terrain_label(data) -> str:
    t = data.get("terrain", {}) or {}
    lab = clean_label(t.get("label_zh"))
    if lab:
        return lab
    if data.get("scene_kind") == "interior":
        return "地板"
    words = (str(t.get("surface", "")) + " " + str((data.get("palette") or {}).get("ground_words", ""))).lower()
    if (data.get("palette") or {}).get("snow_cover") or "snow" in words:
        return "雪地"
    if any(k in words for k in ("sand", "dune", "desert")):
        return "沙丘" if (t.get("bumps") or t.get("ridges")) else "沙地"
    if any(k in words for k in ("grass", "meadow", "lawn")):
        return "草地"
    if any(k in words for k in ("asphalt", "paving", "concrete", "cobble")):
        return "路面"
    return "地面"


def element_label(kind, item, interior=False) -> str:
    """Chinese block name for one JSON element (``kind``: primitive / road / band / foliage / cliff)."""
    item = item or {}
    lab = clean_label(item.get("label_zh"))
    if lab:
        return lab
    try:  # older analyses without label_zh: English ids ("coffee_table", "left_wall") → the massing vocabulary
        from . import massing as _MS
        words = _MS.hint_from_name(str(item.get("id", "")))
    except Exception:
        words = ""
    words = clean_label(words.split("，")[0]) if words else ""
    if words and words not in ("左", "右"):
        return words
    if kind == "primitive":
        t = item.get("type", "")
        return (INTERIOR_TYPE_ZH.get(t) if interior else None) or TYPE_ZH.get(t, "体块")
    if kind == "road":
        return ROAD_ZH.get(item.get("kind", ""), "道路")
    if kind == "band":
        return BAND_ZH.get(item.get("kind", ""), "远山")
    if kind == "foliage":
        return FOLIAGE_ZH.get(item.get("kind", ""), "树")
    if kind == "cliff":
        return "崖壁"
    if kind == "ribbon":
        return "连续曲面"
    return "体块"


def scene_note(data) -> str:
    """One Chinese line for the generation prompt: season / colour lock + scene words (no photo)."""
    p = data.get("palette") or {}
    lock = str(p.get("lock_zh", "") or "").strip()
    env = str(data.get("environment_zh", "") or "").strip()
    if not lock:
        season = SEASON_ZH.get(str(p.get("season", "")), "")
        if p.get("snow_cover"):
            season += "，地面有积雪"
        elif str(p.get("season", "")) not in ("autumn", "winter", "not_applicable", "") and not p.get("autumn_colours"):
            season += "，没有秋季黄叶"
        bits = [b for b in (season,
                            ("地面：" + str(p.get("ground_words", "")).strip()) if p.get("ground_words") else "",
                            ("植被：" + str(p.get("foliage_words", "")).strip()) if p.get("foliage_words") else "",
                            ("天空：" + str(p.get("sky_words", "")).strip()) if p.get("sky_words") else "") if b]
        lock = "；".join(bits)
    try:   # 1.2.6 只做场景相关物体: no pedestrians / traffic words reach the image prompt
        _SS = _static_scene()
        env, lock = _SS.scrub(env).strip(), _SS.scrub(lock).strip()
    except Exception:
        pass
    # Keep the defining connected surfaces before optional atmosphere prose.
    # A long environment sentence must not silently erase the topology cue.
    ribbons = data.get("ribbons") or []
    topology = ""
    if ribbons:
        names = "、".join(dict.fromkeys(element_label("ribbon", r) for r in ribbons[:12]))
        topology = f"连续曲面：{names}，按已有网格保持带状连通、转折和结构间空隙"
    parts = [x for x in (topology, env[:200], lock[:220]) if x]
    if not parts:
        return ""
    return "场景（按环境参考图分析的季节与色彩，保持一致）：" + "；".join(parts)



def _static_scene():
    """1.2.6 只做场景相关物体 helpers (works when this module is loaded by path, as in the tests)."""
    try:
        from . import static_scene as SS
    except ImportError:
        import importlib.util as _ilu
        import os as _os
        import sys as _sys
        SS = _sys.modules.get("wondful_static_scene")
        if SS is None:
            _sp = _ilu.spec_from_file_location("wondful_static_scene",
                                               _os.path.join(_os.path.dirname(__file__), "static_scene.py"))
            SS = _ilu.module_from_spec(_sp)
            _sp.loader.exec_module(SS)
            _sys.modules["wondful_static_scene"] = SS
    return SS


# ---------------------------------------------------------------------------------------- normalise

def _f(v, d=0.0):
    try:
        v = float(v)
        return v if math.isfinite(v) else d
    except (TypeError, ValueError):
        return d


def reference_subject(data):
    """Validate optional image anchors; malformed/legacy input never activates a lock."""
    raw = (data or {}).get("reference_subject") or {}
    empty = {"present": False, "bbox": {k: 0.0 for k in ("x_min", "y_min", "x_max", "y_max")},
             "ground_contact": {"x_frac": 0.5, "y_frac": 0.8}, "heading_deg": 0.0, "heading_confidence": 0.0}
    if not isinstance(raw, dict) or raw.get("present") is not True:
        return empty
    try:
        bb = {k: float(raw["bbox"][k]) for k in empty["bbox"]}
        contact = {k: float(raw["ground_contact"][k]) for k in ("x_frac", "y_frac")}
        values = list(bb.values()) + list(contact.values())
        if not all(math.isfinite(v) for v in values):
            return empty
        if not (0 <= bb["x_min"] < bb["x_max"] <= 1 and 0 <= bb["y_min"] < bb["y_max"] <= 1
                and bb["x_max"] - bb["x_min"] >= 0.01 and bb["y_max"] - bb["y_min"] >= 0.01
                and 0 <= contact["x_frac"] <= 1 and 0 <= contact["y_frac"] <= 1):
            return empty
    except (KeyError, TypeError, ValueError):
        return empty
    return {"present": True, "bbox": bb, "ground_contact": contact,
            "heading_deg": (_f(raw.get("heading_deg")) + 180.0) % 360.0 - 180.0,
            "heading_confidence": max(0.0, min(1.0, _f(raw.get("heading_confidence"))))}


def normalize_ribbons(items):
    """Bound generated geometry and reject invalid paired edges instead of inventing points."""
    clean, warnings = [], []
    for index, item in enumerate(items[:12] if isinstance(items, list) else []):
        if not isinstance(item, dict):
            warnings.append(f"曲面 {index + 1} 无有效截面，已跳过")
            continue
        name = clean_label(item.get("label_zh"), "连续曲面")
        sections = []
        reason = ""
        raw_sections = item.get("sections")
        if not isinstance(raw_sections, list) or not 2 <= len(raw_sections) <= 32:
            reason = "截面数量须为 2–32"
        else:
            for row in raw_sections:
                try:
                    pair = np.array([[float(row[edge][k]) for k in ("x", "y", "z")]
                                     for edge in ("left", "right")], dtype=float)
                    if not np.isfinite(pair).all() or np.max(np.abs(pair)) > 20000:
                        raise ValueError("截面坐标无效")
                    if np.linalg.norm(pair[1] - pair[0]) < 0.02:
                        raise ValueError("截面两侧重合")
                    if sections:
                        previous = sections[-1]
                        if np.linalg.norm(pair - previous) < 1e-5:
                            continue  # harmless duplicate endpoint
                        if np.linalg.norm(pair.mean(0) - previous.mean(0)) < 1e-4:
                            raise ValueError("相邻截面中心重合")
                        # A sudden edge swap creates a bow-tie, not a smooth twist.
                        a, b = previous[1] - previous[0], pair[1] - pair[0]
                        if np.dot(a, b) < -0.5 * np.linalg.norm(a) * np.linalg.norm(b):
                            raise ValueError("相邻截面翻转过大，请增加中间截面")
                    sections.append(pair)
                except (KeyError, TypeError, ValueError, IndexError) as exc:
                    reason = str(exc) or "截面坐标无效"
                    break
        if reason or len(sections) < 2:
            warnings.append(f"{name}：{reason or '有效截面不足'}，已跳过")
            continue
        thickness = max(0.01, min(10.0, _f(item.get("thickness_m"), 0.2)))
        clean.append({"id": str(item.get("id") or f"ribbon_{index + 1}")[:80], "label_zh": name,
                      "sections": [{edge: {k: float(pair[j, axis]) for axis, k in enumerate(("x", "y", "z"))}
                                    for j, edge in enumerate(("left", "right"))} for pair in sections],
                      "thickness_m": thickness, "material": str(item.get("material") or "")[:600]})
    return clean, warnings


def normalize(data) -> dict:
    """Fill defaults so older (prototype v3 / v4) JSONs and partial replies build safely; clamp camera."""
    if not isinstance(data, dict):
        raise ValueError("分析结果不是 JSON 对象")
    d = copy.deepcopy(data)
    d.setdefault("scene_kind", "outdoor")
    if d["scene_kind"] not in ("outdoor", "interior"):
        d["scene_kind"] = "outdoor"
    cam = d.setdefault("camera", {})
    cam["height_m"] = max(0.2, min(500.0, _f(cam.get("height_m"), 1.6)))
    cam["pitch_deg"] = max(-80.0, min(60.0, _f(cam.get("pitch_deg"), 0.0)))
    cam["roll_deg"] = max(-30.0, min(30.0, _f(cam.get("roll_deg"), 0.0)))
    cam["hfov_deg"] = max(20.0, min(120.0, _f(cam.get("hfov_deg"), 60.0)))
    cam["horizon_y_frac"] = _f(cam.get("horizon_y_frac"), 0.5)
    t = d.setdefault("terrain", {})
    t.setdefault("far_m", 500.0)
    t.setdefault("base_slope_deg", 0.0)
    t.setdefault("base_slope_dir_deg", 0.0)
    t.setdefault("smoothness", 0.5)
    t.setdefault("surface", "")
    t.setdefault("bumps", [])
    t.setdefault("ridges", [])
    t.setdefault("has_sea", False)
    t.setdefault("sea_level_m", 0.0)
    for key in ("roads", "mountain_bands", "foliage_masses", "primitives", "cliffs", "surface_zones"):
        if not isinstance(d.get(key), list):
            d[key] = []
    for p in d["primitives"]:
        p.setdefault("id", "p")
        if p.get("type") not in PRIM_TYPES:
            p["type"] = "building"
        if p.get("shape") not in PRIM_SHAPES:
            p["shape"] = "box"
        for k, dv in (("x", 0.0), ("y", 10.0), ("z_offset", 0.0), ("w", 1.0), ("d", 1.0), ("h", 1.0), ("roof_h", 0.0),
                      ("rot_z_deg", 0.0), ("repeat_dx", 0.0), ("repeat_dy", 0.0)):
            p[k] = _f(p.get(k), dv)
        try:
            p["repeat"] = int(p.get("repeat", 1) or 1)
        except (TypeError, ValueError):
            p["repeat"] = 1
    spot = d.setdefault("product_spot", {})
    spot["x"] = _f(spot.get("x"), 0.0)
    spot["y"] = _f(spot.get("y"), 8.0)
    spot["rot_z_deg"] = _f(spot.get("rot_z_deg"), 0.0)
    d["reference_subject"] = reference_subject(d)
    d["ribbons"], warnings = normalize_ribbons(d.get("ribbons"))
    if warnings:
        d["_geometry_warnings"] = list(dict.fromkeys(list(d.get("_geometry_warnings") or []) + warnings))
    d.setdefault("palette", {})
    return d


def frame_of(data, default=(1600, 900)):
    fr = (data or {}).get("_frame") or {}
    w, h = int(_f(fr.get("w"), default[0]) or default[0]), int(_f(fr.get("h"), default[1]) or default[1])
    return max(1, w), max(1, h)


def lens_from_hfov(hfov_deg, sensor=36.0) -> float:
    return (sensor / 2.0) / math.tan(math.radians(max(20.0, min(120.0, hfov_deg))) / 2.0)


def horizon_from_pitch(pitch_deg, hfov_deg, hw) -> float:
    tv = math.tan(math.radians(hfov_deg) / 2.0) * hw
    return 0.5 + math.tan(math.radians(pitch_deg)) / (2.0 * tv)


def yaw_to(dx, dy) -> float:
    """Rotation about Z (radians) that turns +Y into the horizontal direction (dx, dy)."""
    return math.atan2(-dx, dy)


def align_yaw(product_axis_angle, spot_heading_deg) -> float:
    """Yaw for the blockout so the spot heading lands on the product's long axis (mod 180°, smallest turn)."""
    a = product_axis_angle - math.radians(spot_heading_deg)
    a = (a + math.pi / 2) % math.pi - math.pi / 2
    return a


def camera_rotation(camera):
    """Camera-to-world matrix matching Blender Euler((90+pitch), roll, 0), XYZ."""
    a, b = math.radians(90 + _f(camera.get("pitch_deg"))), math.radians(_f(camera.get("roll_deg")))
    ca, sa, cb, sb = math.cos(a), math.sin(a), math.cos(b), math.sin(b)
    return np.array([[cb, sb * sa, sb * ca], [0.0, ca, -sa], [-sb, cb * sa, cb * ca]], float)


def project_points(camera, frame, points, camera_ground=0.0):
    """Normalized screen points, or None when any point is behind the near plane."""
    pts = np.asarray(points, float)
    if pts.ndim != 2 or pts.shape[1] != 3 or not np.isfinite(pts).all():
        return None
    origin = np.array([0.0, 0.0, _f(camera.get("height_m"), 1.6) + camera_ground])
    q = (pts - origin) @ camera_rotation(camera)
    depth = -q[:, 2]
    if np.any(depth <= 0.05):
        return None
    tx = math.tan(math.radians(_f(camera.get("hfov_deg"), 60)) / 2.0)
    ty = tx * frame[1] / float(frame[0])
    return np.column_stack((0.5 + q[:, 0] / (2 * depth * tx), 0.5 - q[:, 1] / (2 * depth * ty)))


def framing_metrics(camera, frame, relative_points, spot, camera_ground=0.0):
    """Actual target bounds and its ground-footprint centre, not the reference car's height."""
    anchor = np.asarray(spot, float)
    projected = project_points(camera, frame, np.vstack((np.asarray(relative_points, float) + anchor, anchor)), camera_ground)
    if projected is None or len(projected) < 2:
        return None
    product, contact = projected[:-1], projected[-1]
    lo, hi = product.min(0), product.max(0)
    return {"bbox": {"x_min": float(lo[0]), "y_min": float(lo[1]), "x_max": float(hi[0]), "y_max": float(hi[1])},
            "width": float(hi[0] - lo[0]), "center_x": float((lo[0] + hi[0]) / 2),
            "ground_contact": {"x_frac": float(contact[0]), "y_frac": float(contact[1])}}


def framing_errors(subject, metrics):
    bb, gc = subject["bbox"], subject["ground_contact"]
    return {"width": metrics["width"] - (bb["x_max"] - bb["x_min"]),
            "center_x": metrics["center_x"] - (bb["x_min"] + bb["x_max"]) / 2,
            "contact_x": metrics["ground_contact"]["x_frac"] - gc["x_frac"],
            "contact_y": metrics["ground_contact"]["y_frac"] - gc["y_frac"]}


def fit_subject_framing(camera, frame, subject, relative_points, initial_xy=(0.0, 8.0), ground_height=None):
    """Fit x/y to measured anchors with a fixed camera and actual target bounds.

    This is an overdetermined 2D fit: differing product dimensions or a wrong camera
    can leave a residual. The caller must retain/report it; target height is never
    matched to the photographed product, and no target geometry is modified.
    """
    if not subject.get("present"):
        return {"enabled": False, "reason": "NO_REFERENCE_SUBJECT"}
    pts = np.asarray(relative_points, float)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 2 or not np.isfinite(pts).all():
        return {"enabled": False, "reason": "NO_TARGET_BOUNDS"}
    ground = ground_height or (lambda x, y: 0.0)
    cg = float(ground(0.0, 0.0))
    weights = np.array([2.0, 1.0, 0.5, 1.0])

    def evaluate(xy):
        try:
            z = float(ground(float(xy[0]), float(xy[1])))
            metrics = framing_metrics(camera, frame, pts, (xy[0], xy[1], z), cg)
            if metrics is None:
                return None
            errors = framing_errors(subject, metrics)
            residual = np.array([errors[k] for k in ("width", "center_x", "contact_x", "contact_y")]) * weights
            return residual, metrics, errors, z
        except (ValueError, TypeError, FloatingPointError):
            return None

    starts = [np.array(initial_xy, float)]
    tx = math.tan(math.radians(camera["hfov_deg"]) / 2)
    ty = tx * frame[1] / float(frame[0])
    gc = subject["ground_contact"]
    ray = camera_rotation(camera) @ np.array([(2 * gc["x_frac"] - 1) * tx, (1 - 2 * gc["y_frac"]) * ty, -1.0])
    if ray[2] < -1e-6:
        distance = camera["height_m"] / -ray[2]
        starts.append(ray[:2] * distance)
    width = subject["bbox"]["x_max"] - subject["bbox"]["x_min"]
    approx_y = max(float(np.ptp(pts[:, 0])) / max(2 * tx * width, 1e-5), float(np.ptp(pts[:, 1])) + 0.5)
    starts.append(np.array([(2 * gc["x_frac"] - 1) * tx * approx_y, approx_y]))
    best = None
    iterations = 0
    min_y = max(0.3, -float(pts[:, 1].min()) + 0.3)
    for initial in starts:
        xy = np.clip(initial, (-20000.0, min_y), (20000.0, 20000.0))
        for _ in range(35):
            iterations += 1
            state = evaluate(xy)
            if state is None:
                xy[1] *= 1.4
                continue
            r = state[0]
            cost = float(r @ r)
            if best is None or cost < best[0]:
                best = (cost, xy.copy(), state)
            if cost < 1e-14:
                break
            columns = []
            for axis in range(2):
                eps = 1e-4 * max(1.0, abs(float(xy[axis])))
                offset = xy.copy()
                offset[axis] += eps
                other = evaluate(offset)
                if other is None:
                    break
                columns.append((other[0] - r) / eps)
            if len(columns) != 2:
                break
            jac = np.column_stack(columns)
            step = np.linalg.lstsq(jac, -r, rcond=None)[0]
            step_norm = float(np.linalg.norm(step))
            if step_norm > max(1.0, xy[1]) * 0.5:
                step *= max(1.0, xy[1]) * 0.5 / step_norm
            improved = False
            for factor in (1.0, 0.5, 0.25, 0.1):
                candidate = np.clip(xy + factor * step, (-20000.0, min_y), (20000.0, 20000.0))
                trial = evaluate(candidate)
                if trial is not None and float(trial[0] @ trial[0]) < cost - 1e-15:
                    xy, improved = candidate, True
                    break
            if not improved:
                break
    if best is None:
        return {"enabled": False, "reason": "NO_VISIBLE_FIT"}
    _, xy, (_, metrics, errors, z) = best
    residual = max(abs(v) for v in errors.values())
    return {"enabled": True, "status": "FITTED" if residual <= 0.025 else "RESIDUAL",
            "x": float(xy[0]), "y": float(xy[1]), "z": float(z), "metrics": metrics,
            "target": copy.deepcopy(subject), "errors": errors, "max_error_frac": float(residual),
            "iterations": iterations, "height_matched": False, "heading_undirected": True}


def ribbon_mesh(sections, thickness_m=0.2, subdivisions=6, width_subdivisions=None, return_layout=False):
    """Closed ruled ribbon with bounded Hermite edges and a tessellated width.

    The legacy two-value return is unchanged. ``return_layout`` additionally
    exposes skin/rim face ranges so Blender can keep their shading normals apart.
    Width tessellation preserves the source edges and thickness offsets exactly;
    it resolves twisted ruled surfaces that one wide quad cannot represent.
    """
    normalized, _warnings = normalize_ribbons([{"sections": sections, "thickness_m": thickness_m}])
    if not normalized:
        raise ValueError("没有可用的连续曲面截面")
    source = normalized[0]
    pairs = np.array([[[row[edge][k] for k in ("x", "y", "z")] for edge in ("left", "right")]
                      for row in source["sections"]], float)
    tangent = np.empty_like(pairs)
    tangent[0], tangent[-1] = pairs[1] - pairs[0], pairs[-1] - pairs[-2]
    tangent[1:-1] = (pairs[2:] - pairs[:-2]) * 0.5
    # Limit tangents at unevenly spaced sections so a short segment cannot overshoot wildly.
    lengths = np.linalg.norm(np.diff(pairs, axis=0).reshape(len(pairs) - 1, -1), axis=1)
    for i in range(1, len(pairs) - 1):
        norm = float(np.linalg.norm(tangent[i]))
        if norm > 0:
            tangent[i] *= min(1.0, min(lengths[i - 1], lengths[i]) / norm)
    count = max(1, min(16, int(subdivisions)))
    sampled = []
    for i in range(len(pairs) - 1):
        for t in np.linspace(0.0, 1.0, count, endpoint=False):
            t2, t3 = t * t, t * t * t
            row = ((2 * t3 - 3 * t2 + 1) * pairs[i] + (t3 - 2 * t2 + t) * tangent[i]
                   + (-2 * t3 + 3 * t2) * pairs[i + 1] + (t3 - t2) * tangent[i + 1])
            if np.linalg.norm(row[1] - row[0]) < 0.02:
                row = (1 - t) * pairs[i] + t * pairs[i + 1]
            sampled.append(row)
    sampled.append(pairs[-1])
    rows = np.array(sampled)
    centres = rows.mean(1)
    along = np.gradient(centres, axis=0)
    normals = np.cross(rows[:, 1] - rows[:, 0], along)
    norms = np.linalg.norm(normals, axis=1)
    if np.any(norms < 1e-7):
        raise ValueError("曲面截面与路径共线，无法形成有效曲面")
    normals /= norms[:, None]
    for i in range(1, len(normals)):
        if normals[i] @ normals[i - 1] < 0:
            normals[i] *= -1
    half = min(source["thickness_m"], float(np.min(np.linalg.norm(rows[:, 1] - rows[:, 0], axis=1))) * 0.5) / 2
    upper, lower = rows + normals[:, None, :] * half, rows - normals[:, None, :] * half

    def grid(steps):
        t = np.linspace(0.0, 1.0, steps + 1)[None, :, None]
        return tuple(edge[:, :1] * (1 - t) + edge[:, 1:] * t for edge in (upper, lower))

    def twist_degrees(surfaces):
        maximum = 0.0
        for surface in surfaces:
            a, b, c, d = surface[:-1, :-1], surface[:-1, 1:], surface[1:, 1:], surface[1:, :-1]
            # Both possible diagonals, so the bound is independent of Blender's
            # triangulation choice for a nonplanar quad.
            for first, second in ((np.cross(b - a, c - a), np.cross(c - a, d - a)),
                                  (np.cross(b - a, d - a), np.cross(c - b, d - b))):
                denom = np.linalg.norm(first, axis=-1) * np.linalg.norm(second, axis=-1)
                cosine = np.sum(first * second, axis=-1) / np.maximum(denom, 1e-20)
                maximum = max(maximum, float(np.degrees(np.arccos(np.clip(cosine, -1, 1))).max()))
        return maximum

    steps = 1 if width_subdivisions is None else max(1, min(64, int(width_subdivisions)))
    surfaces = grid(steps)
    twist = twist_degrees(surfaces)
    if width_subdivisions is None:
        while twist > 6.0 and steps < 64:
            steps *= 2
            surfaces = grid(steps)
            twist = twist_degrees(surfaces)
    columns, stride = steps + 1, (steps + 1) * 2
    vertices = np.concatenate(surfaces, axis=1).reshape(-1, 3)
    faces = []
    for i in range(len(rows) - 1):
        a, b = stride * i, stride * (i + 1)
        for j in range(steps):
            faces.extend(((a + j, a + j + 1, b + j + 1, b + j),
                          (a + columns + j, b + columns + j, b + columns + j + 1, a + columns + j + 1)))
    skin_faces = len(faces)
    for i in range(len(rows) - 1):
        a, b = stride * i, stride * (i + 1)
        faces.extend(((a, b, b + columns, a + columns),
                      (a + steps, a + columns + steps, b + columns + steps, b + steps)))
    end = stride * (len(rows) - 1)
    for j in range(steps):
        faces.extend(((j, columns + j, columns + j + 1, j + 1),
                      (end + j, end + j + 1, end + columns + j + 1, end + columns + j)))
    corners = vertices[np.array(faces)]
    area = np.linalg.norm(np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1)
    area += np.linalg.norm(np.cross(corners[:, 2] - corners[:, 0], corners[:, 3] - corners[:, 0]), axis=1)
    if not np.isfinite(vertices).all() or np.any(area < 1e-8):
        raise ValueError("曲面插值产生退化面，请调整截面")
    layout = {"rows": len(rows), "columns": columns, "width_subdivisions": steps,
              "skin_faces": skin_faces, "max_quad_twist_deg": twist}
    return (vertices, faces, layout) if return_layout else (vertices, faces)


# ---------------------------------------------------------------------------------------- product spot

NON_BLOCKING = ("rug", "ceiling", "water_plane")


def rect_dist(px, py, p) -> float:
    """Distance from (px, py) to a primitive's rotated footprint (0 inside)."""
    rz = math.radians(_f(p.get("rot_z_deg"), 0.0))
    dx, dy = px - _f(p.get("x")), py - _f(p.get("y"))
    lx = dx * math.cos(rz) + dy * math.sin(rz)
    ly = -dx * math.sin(rz) + dy * math.cos(rz)
    ex = max(0.0, abs(lx) - max(_f(p.get("w"), 1.0), 0.05) / 2)
    ey = max(0.0, abs(ly) - max(_f(p.get("d"), 1.0), 0.05) / 2)
    return math.hypot(ex, ey)


def blocking_footprints(prims, height):
    """Footprints standing where the product would stand (repeats expanded; rugs / ceilings / raised items skipped)."""
    out = []
    for p in prims or ():
        if p.get("type") in NON_BLOCKING or _f(p.get("z_offset"), 0.0) >= height:
            continue
        n = int(max(1, min(40, int(_f(p.get("repeat"), 1) or 1))))
        for k in range(n):
            out.append(dict(p, x=_f(p.get("x")) + k * _f(p.get("repeat_dx")), y=_f(p.get("y")) + k * _f(p.get("repeat_dy"))))
    return out


def clearance(prims, x, y) -> float:
    return min((rect_dist(x, y, p) for p in prims), default=1e9)


def free_spot(prims, x, y, radius, height, step=0.1, max_shift=None, hfov_deg=None):
    """Nearest product spot (within ``max_shift``, default max(2.5 × radius, 1.2 m)) whose footprint circle is clear
    of standing primitives and, with ``hfov_deg``, still in the middle 80 % of the frame width. Returns
    (x, y, shift_m); shift 0 when the analysed spot is already free, None when nothing nearby is free (the builder
    then removes the overlapping blocks)."""
    obs = blocking_footprints(prims, height)
    if clearance(obs, x, y) >= radius:
        return x, y, 0.0
    max_shift = max(2.5 * radius, 1.2) if max_shift is None else max_shift
    tanh = math.tan(math.radians(hfov_deg) / 2) if hfov_deg else None

    def visible(cx, cy):
        return tanh is None or (cy > 0.5 and 0.1 <= 0.5 + cx / (2 * cy * tanh) <= 0.9)
    k = 1
    while k * step <= max_shift + 1e-9:
        r = k * step
        best = None
        for j in range(24):
            a = 2 * math.pi * j / 24
            cx, cy = x + r * math.sin(a), y + r * math.cos(a)
            c = clearance(obs, cx, cy)
            if c >= radius and visible(cx, cy) and (best is None or c > best[2]):
                best = (cx, cy, c)
        if best is not None:
            return best[0], best[1], r
        k += 1
    return None


# ---------------------------------------------------------------------------------------- analysis call

def product_words(vehicle, dims) -> str:
    if vehicle:
        return "a car"
    w, d, h = dims
    if max(w, d, h) < 1.5:
        return "a small object / piece of furniture"
    return "an object"


def analysis_content(ref_url, frame, dims=(4.8, 1.9, 1.5), vehicle=True):
    fw, fh = frame
    hw = fh / float(fw)
    g = math.gcd(int(fw), int(fh)) or 1
    aspect_txt = f"{fw // g}:{fh // g}" if max(fw // g, fh // g) <= 40 else f"{fw / fh:.3f}:1"
    pw, pd, ph = dims
    rules = RULES.format(fw=fw, fh=fh, aspect_txt=aspect_txt, hw=hw, product_words=product_words(vehicle, dims),
                         pw=pw, pd=pd, ph=ph)
    return [{"type": "input_text", "text": rules},
            {"type": "input_text", "text": f"Reference photo ({fw}x{fh}):"},
            {"type": "input_image", "image_url": ref_url}]


def correction_content(base_content, prev, scores, hints, render_url, overlay_url):
    prevj = {k: v for k, v in prev.items() if not k.startswith("_")}
    return list(base_content) + [
        {"type": "input_text", "text": CORRECTION.format(scores=json.dumps(scores), hints="\n".join(hints) or "(none)",
                                                         prev=json.dumps(prevj, ensure_ascii=False))},
        {"type": "input_text", "text": "Clay render of the previous JSON:"},
        {"type": "input_image", "image_url": render_url},
        {"type": "input_text", "text": "50% overlay (render over photo):"},
        {"type": "input_image", "image_url": overlay_url},
    ]


def payload(content, model, effort):
    return {"model": model,
            "instructions": "Reconstruct scene layout as strict JSON. Do not use any tool.",
            "input": [{"type": "message", "role": "user", "content": content}],
            "tools": [], "parallel_tool_calls": False, "store": False, "stream": True, "include": [],
            "reasoning": {"effort": effort},
            "text": {"format": {"type": "json_schema", "name": SCHEMA_NAME, "strict": True, "schema": SCHEMA}}}


def parse_reply(text) -> dict:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        raise ValueError("分析没有返回 JSON")
    return json.loads(m.group(0))


def call_analysis(content, *, effort=EFFORT_FIRST, model="", client_version="", cancel_check=None, timeout=900) -> dict:
    """One strict-JSON reasoning call with the user's Codex login (codex_direct auth / SSE)."""
    import os
    from . import codex_direct as CD
    auth_path = CD.find_auth_file()
    if auth_path is None:
        raise CD.DirectEditError("未找到 Codex 登录文件，请先在 ① 连接 Codex", kind="auth_local", fatal=True)
    auth = CD.load_auth(auth_path)
    if CD.token_expiring(CD.access_token(auth)):
        auth = CD.refresh_auth(auth, auth_path)
    base = os.environ.get(CD.BASE_URL_ENV, "").strip() or CD.DEFAULT_BASE_URL
    body = payload(content, (model or "").strip() or CD.DEFAULT_MODEL, effort)
    refreshed = dropped = False
    while True:
        try:
            text = CD.post_responses(base.rstrip("/") + "/responses", CD.auth_headers(auth, client_version), body,
                                     timeout=timeout, cancel_check=cancel_check, want="text")
            break
        except CD.DirectEditError as exc:
            if exc.kind == "auth" and exc.status == 401 and not refreshed:
                refreshed = True
                auth = CD.refresh_auth(auth, auth_path)
                continue
            if exc.kind in ("network", "stream_cut") and not exc.fatal and not dropped and not (cancel_check and cancel_check()):
                dropped = True  # one retry on a dropped stream, never more
                continue
            raise
    return parse_reply(text)


# ---------------------------------------------------------------------------------------- fit score (numpy)

def _resize_nn(a, w, h):
    H, W = a.shape[:2]
    ys = np.clip((np.arange(h) + 0.5) * H / h, 0, H - 1).astype(int)
    xs = np.clip((np.arange(w) + 0.5) * W / w, 0, W - 1).astype(int)
    return a[ys][:, xs]


def _box(a, r=1):
    p = np.pad(a, r, mode="edge")
    out = np.zeros_like(a, dtype=np.float32)
    for dy in range(2 * r + 1):
        for dx in range(2 * r + 1):
            out += p[dy:dy + a.shape[0], dx:dx + a.shape[1]]
    return out / float((2 * r + 1) ** 2)


def _dilate(m, r):
    out = m.copy()
    p = np.pad(m, r)
    for dy in range(2 * r + 1):
        for dx in range(2 * r + 1):
            out |= p[dy:dy + m.shape[0], dx:dx + m.shape[1]]
    return out


def photo_sky(rgb) -> np.ndarray:
    """Sky mask of a photo (float RGB 0..1): bright, low-saturation or blue pixels connected to the top edge."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = rgb.max(-1)
    mn = rgb.min(-1)
    sat = (mx - mn) / np.maximum(mx, 1e-4)
    cand = (mx > 0.5) & ((sat < 0.22) | ((b > r + 0.04) & (b >= g - 0.03)))
    cand[int(cand.shape[0] * 0.85):] = False
    sky = np.zeros_like(cand)
    sky[0] = cand[0]
    for _ in range(cand.shape[0] * 2):
        grow = _dilate(sky, 1) & cand
        if (grow == sky).all():
            break
        sky = grow
    return sky


def edges(lum, q=0.85, valid=None) -> np.ndarray:
    s = _box(lum.astype(np.float32), 1)
    gy, gx = np.gradient(s)
    mag = np.hypot(gx, gy)
    samples = mag[valid] if valid is not None else mag
    t = float(np.quantile(samples, q)) if samples.size else 0.0
    result = mag > max(t, 1e-3)
    return result & valid if valid is not None else result


def fit_score(photo_rgb, render_rgba, exclude_bbox=None) -> dict:
    """Numeric blockout fit (0..1) of a clay render against the photo.

    sky IoU (photo sky heuristic vs render background alpha) when either has sky, plus the edge F-score
    (2 px tolerance) of luminance edges; ``hints`` = skyline differences per tenth of the frame.
    Optional normalized subject bbox is excluded because this preview contains only the environment.
    """
    H0, W0 = photo_rgb.shape[:2]
    w = SCORE_W
    h = max(8, int(round(w * H0 / float(W0))))
    ph = _resize_nn(photo_rgb[..., :3], w, h).astype(np.float32)
    rr = _resize_nn(render_rgba, w, h).astype(np.float32)
    if ph.max() > 1.5:
        ph = ph / 255.0
    if rr.max() > 1.5:
        rr = rr / 255.0
    alpha = rr[..., 3] if rr.shape[-1] == 4 else np.ones(rr.shape[:2], np.float32)
    rsky = alpha < 0.5
    psky = photo_sky(ph)
    valid = np.ones((h, w), bool)
    if exclude_bbox is not None:
        try:
            box = ([exclude_bbox[k] for k in ("x_min", "y_min", "x_max", "y_max")]
                   if isinstance(exclude_bbox, dict) else list(exclude_bbox))
            x0, y0, x1, y1 = (float(v) for v in box)
            if all(math.isfinite(v) for v in (x0, y0, x1, y1)) and 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1:
                valid[max(0, int(y0 * h) - 3):min(h, math.ceil(y1 * h) + 3),
                      max(0, int(x0 * w) - 3):min(w, math.ceil(x1 * w) + 3)] = False
        except (ValueError, TypeError, KeyError):
            pass
    count = int(valid.sum())
    if not count:
        return {"score": 0.0, "sky_photo": 0.0, "sky_render": 0.0, "edge_f": 0.0,
                "hints": [], "excluded_fraction": 1.0, "reason": "NO_ENVIRONMENT_PIXELS"}
    sky_photo = float((psky & valid).sum()) / count
    sky_render = float((rsky & valid).sum()) / count
    out = {"sky_photo": round(sky_photo, 3), "sky_render": round(sky_render, 3)}
    if exclude_bbox is not None:
        out["excluded_fraction"] = round(1 - count / float(h * w), 3)
    use_sky = sky_photo > 0.02 or sky_render > 0.02
    if use_sky:
        inter, union = float((psky & rsky & valid).sum()), float(((psky | rsky) & valid).sum())
        out["sky_iou"] = round(inter / union, 3) if union else 1.0
    lp = ph @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    lr = rr[..., :3] @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    lr = np.where(rsky, 1.0, lr)
    ep, er = edges(lp, valid=valid), edges(lr, valid=valid)
    prec = float((er & _dilate(ep, 2)).sum()) / max(1.0, float(er.sum()))
    rec = float((ep & _dilate(er, 2)).sum()) / max(1.0, float(ep.sum()))
    out["edge_f"] = round(2 * prec * rec / max(1e-6, prec + rec), 3)
    out["score"] = round(0.6 * out["sky_iou"] + 0.4 * out["edge_f"], 3) if use_sky else out["edge_f"]
    hints = []
    if use_sky:
        for k in range(10):
            c0, c1 = int(k * w / 10), int((k + 1) * w / 10)

            def top(m):
                known = valid[:, c0:c1]
                n = known.sum(1)
                col = (((~m[:, c0:c1]) & known).sum(1) > n * 0.5) & (n > 0)
                idx = np.nonzero(col)[0]
                return float(idx[0]) / h if idx.size else 1.0
            a, b = top(psky), top(rsky)
            if abs(a - b) > 0.04:
                hints.append(f"x {k / 10:.1f}-{(k + 1) / 10:.1f}: photo skyline at y {a:.2f}, blockout at y {b:.2f} "
                             f"(blockout {'too high' if b < a else 'too low'})")
    out["hints"] = hints
    return out


def overlay(photo_rgb, render_rgb) -> np.ndarray:
    H, W = photo_rgb.shape[:2]
    rr = _resize_nn(render_rgb[..., :3], W, H).astype(np.float32)
    return (0.5 * photo_rgb[..., :3].astype(np.float32) + 0.5 * rr).clip(0, 255).astype(np.uint8)


def summary(info) -> str:
    """Panel note after a build."""
    if not info:
        return ""
    bits = [f"场景粗模：{info.get('blocks', 0)} 个体块"]
    if info.get("kind") == "interior":
        bits.append("室内")
    if info.get("camera"):
        bits.append("已设置「" + CAMERA_NAME + "」")
    sc = info.get("scores") or []
    if len(sc) >= 2:
        bits.append(f"吻合度 {sc[0]:.3f}→{sc[1]:.3f}" + ("（采用校正轮）" if info.get("round") == 2 else "（没有明显提高，保留第 1 轮）"))
    elif len(sc) == 1:
        bits.append(f"吻合度 {sc[0]:.3f}")
    if info.get("cached"):
        bits.append("用已保存的分析")
    pr = info.get("precise")
    if pr:
        c = pr.get("camera") or {}
        txt = "精确模式"
        if c.get("hfov_deg") is not None:
            txt += f"（实测视角 {float(c['hfov_deg']):.0f}°）"
        if pr.get("err_after_pct") is not None:
            txt += f" · 3D 距离误差 {pr['err_after_pct']}%"
            if pr.get("err_before_pct") is not None:
                txt += f"（对齐前 {pr['err_before_pct']}%）"
        bits.append(txt)
    note = " · ".join(bits)
    warn = [w for w in info.get("notes", []) if w]
    if info.get("precise_fallback"):
        warn.insert(0, "精确模式不可用，已用普通模式（" + str(info["precise_fallback"])[:60] + "）")
    if warn:
        note += " · 注意：" + "；".join(warn[:2])
    return note
