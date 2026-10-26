"""Non-destructive version management for reference blockouts.

A new build owns a new collection. Older collections and their data are never
removed by generation; they are only hidden from viewport/render so that later
passes cannot double-count them. This module has no import-time Blender deps.
"""
from __future__ import annotations

import re

TAG = 'wondful_scene_blockout'
VERSION = 'wondful_sb_version'
ACTIVE = 'wondful_sb_active'


def all_versions(scene):
    return [c for c in getattr(scene.collection, 'children_recursive', ()) if c.get(TAG)]


def next_number(scene, base_name='Wondful 场景粗模'):
    maximum = 0
    for coll in all_versions(scene):
        try:
            maximum = max(maximum, int(coll.get(VERSION, 0)))
        except (TypeError, ValueError):
            pass
        match = re.search(r'\b(\d{3,})$', str(getattr(coll, 'name', '')))
        if match and base_name in coll.name:
            maximum = max(maximum, int(match.group(1)))
    return maximum + 1


def version_name(number, base_name='Wondful 场景粗模'):
    return f'{base_name} · {int(number):03d}'


def active(scene, props=None):
    """Most recent visible owned blockout, never an unrelated user collection."""
    versions = all_versions(scene)
    picked = getattr(props, 'sb_active_collection', None) if props is not None else None
    if picked in versions and picked.get(TAG) and not picked.hide_render:
        return picked
    visible = [c for c in versions if not getattr(c, 'hide_render', False)]
    tagged = [c for c in visible if c.get(ACTIVE)]
    candidates = tagged or visible
    return max(candidates, key=lambda c: int(c.get(VERSION, 0) or 0)) if candidates else None


def _suspend(coll):
    """Keep objects and their previous user visibility flags for restoration."""
    coll['wondful_sb_saved_hide_render'] = int(bool(coll.hide_render))
    coll['wondful_sb_saved_hide_viewport'] = int(bool(getattr(coll, 'hide_viewport', False)))
    coll.hide_render = True
    if hasattr(coll, 'hide_viewport'):
        coll.hide_viewport = True
    for obj in list(getattr(coll, 'all_objects', ())):
        if getattr(obj, 'type', '') in {'CAMERA', 'LIGHT'}:
            continue
        obj['wondful_sb_saved_render'] = int(bool(obj.hide_render))
        obj.hide_render = True
        if hasattr(obj, 'hide_get') and hasattr(obj, 'hide_set'):
            try:
                obj['wondful_sb_saved_viewport'] = int(bool(obj.hide_get()))
                obj.hide_set(True)
            except (RuntimeError, ReferenceError):
                pass
    coll[ACTIVE] = 0


def activate(scene, new_collection, props=None):
    """Activate the successful build, retaining every prior version in Outliner."""
    versions = all_versions(scene)
    if new_collection not in versions or not new_collection.get(TAG):
        raise ValueError('Not a generated blockout in this scene')
    for coll in versions:
        if coll is new_collection:
            continue
        if not coll.hide_render or coll.get(ACTIVE):
            _suspend(coll)
    new_collection.hide_render = False
    if hasattr(new_collection, 'hide_viewport'):
        new_collection.hide_viewport = False
    for obj in list(getattr(new_collection, 'all_objects', ())):
        if getattr(obj, 'type', '') in {'CAMERA', 'LIGHT'}:
            continue
        if 'wondful_sb_saved_render' in obj:
            obj.hide_render = bool(obj['wondful_sb_saved_render'])
        if 'wondful_sb_saved_viewport' in obj and hasattr(obj, 'hide_set'):
            try:
                obj.hide_set(bool(obj['wondful_sb_saved_viewport']))
            except (RuntimeError, ReferenceError):
                pass
    new_collection[ACTIVE] = 1
    if props is not None and hasattr(props, 'sb_active_collection'):
        props.sb_active_collection = new_collection
    return new_collection
