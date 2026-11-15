"""Shared, quiet panel primitives. No rendering/network/file IO during draw."""
import unicodedata

INPUT_HEIGHT = 1.2
ACTION_HEIGHT = 1.35
PREVIEW_MAX = 14.0

HELP = {
    'environment': '\u4f7f\u7528\u5f53\u524d\u76f8\u673a\u6216\u89c6\u53e3\u6355\u83b7\u753b\u9762\u4f5c\u4e3a\u5e95\u56fe\uff0c\u4e0d\u81ea\u52a8\u751f\u6210\u767d\u6a21\uff1b\u73af\u5883\u53c2\u8003\u63d0\u4f9b\u6750\u8d28\u3001\u5149\u7167\u548c\u6c1b\u56f4\u3002',
    'reference': '\u5355\u56fe\u4f7f\u7528\u76ee\u6807\u539f\u56fe\u3001\u73b0\u6709\u76f8\u673a\u4e0b\u7684\u771f\u5b9e\u4ea7\u54c1\u7ed3\u6784\u6295\u5f71\u548c\u8eab\u4efd\u56fe\uff0c\u81ea\u52a8 AI \u6da6\u8272\u540e\u7f16\u8f91\u539f\u56fe\uff1b\u4e0d\u751f\u6210\u573a\u666f\u7c97\u6a21\u3002\u591a\u89d2\u5ea6\u7c97\u6a21\u5de5\u5177\u4f4d\u4e8e\u9ad8\u7ea7\u2192\u591a\u89d2\u5ea6\u3002',
    'PRODUCT': '\u5fc5\u586b\u4ea7\u54c1\u96c6\u5408\u3001\u73b0\u6709\u6d3b\u52a8\u76f8\u673a\u548c\u4e00\u5f20\u4ea7\u54c1\u8eab\u4efd\u53c2\u8003\uff0c\u53ef\u7528\u6b63\u3001\u4fa7\u3001\u540e\u4e09\u89c6\u56fe\u62fc\u56fe\u3002\u771f\u5b9e\u6a21\u578b\u6295\u5f71\u552f\u4e00\u51b3\u5b9a\u53ef\u89c1\u51e0\u4f55\u3001\u8f66\u578b\u3001\u706f\u7ec4\u3001\u8f6e\u8f90\u548c\u8f66\u6f06\u8272\u76f8\uff1b\u4e09\u89c6\u56fe\u53ea\u6838\u5bf9\u5404\u90e8\u4ef6\u6750\u8d28\uff0c\u4e0d\u6539\u8f66\u6f06\u989c\u8272\u3002\u73af\u5883\u53ea\u6539\u53cd\u5c04\u548c\u53d7\u5149\uff1b\u4ec5\u7528\u6237\u660e\u786e\u586b\u5199\u5177\u4f53\u8f66\u6f06\u989c\u8272\u65f6\u624d\u53ef\u8986\u76d6\u3002',
    'STYLE': '\u63d0\u4f9b\u73af\u5883\u5916\u89c2\u548c\u5149\u7167\uff0c\u4e0d\u63d0\u4f9b\u673a\u4f4d\u6216\u8f66\u578b\u3002\u53ef\u9009\u591a\u5f20\uff0c\u7b2c\u4e00\u5f20\u4e3a\u4e3b\u53c2\u8003\uff1b\u201c\u6dfb\u52a0\u201d\u4e0d\u4f1a\u8986\u76d6\u5df2\u6709\u56fe\u7247\u3002',
    'TARGET': '\u590d\u523b\u76ee\u6807\u539f\u56fe\uff0c\u4e0e\u73af\u5883\u53c2\u8003\u72ec\u7acb\u3002\u6362\u56fe\u540e\u65e7\u7c97\u6a21\u4e0d\u4f1a\u5220\u9664\uff1b\u4ec5\u591a\u89d2\u5ea6\u9700\u8981\u91cd\u65b0\u751f\u6210\u5339\u914d\u65b0\u76ee\u6807\u7684\u7c97\u6a21\u3002',
    'PERSON': '\u4eba\u7269\u5916\u89c2\u53c2\u8003\uff0c\u4e0d\u7528\u4e8e\u8f66\u578b\u8eab\u4efd\u6216\u76f8\u673a\u5339\u914d\u3002',
    'preview': '\u5237\u65b0\u4ec5\u6e32\u67d3\u5f53\u524d\u76f8\u673a\u4e0e\u7c97\u6a21\uff0c\u4e0d\u91cd\u65b0\u5206\u6790\u3002\u6309\u610f\u89c1\u4fee\u6539\u4f1a\u8c03\u7528\u5728\u7ebf\u5206\u6790\uff0c\u4ec5\u4fee\u6539\u63d2\u4ef6\u7c97\u6a21\u548c\u76f8\u673a\uff0c\u4e0d\u6539\u8f66\u8eab\u7f51\u683c\u6216\u6750\u8d28\u3002',
    'prompt': '\u5b8c\u6574\u63d0\u793a\u8bcd\u4f18\u5148\uff0c\u672a\u586b\u65f6\u4f7f\u7528\u4e00\u53e5\u8bdd\u9700\u6c42\u3002\u6bcf\u6b21\u751f\u6210\u524d\u5747\u81ea\u52a8\u8fdb\u884c AI \u6da6\u8272\uff1b\u4e24\u79cd\u6a21\u5f0f\u72ec\u7acb\u4fdd\u5b58\u8349\u7a3f\u3002',
    'build': '\u5728\u9ad8\u7ea7\u2192\u591a\u89d2\u5ea6\uff0c\u6839\u636e\u76ee\u6807\u56fe\u751f\u6210\u76f8\u673a\u548c\u7c97\u6a21\u3002\u6bcf\u6b21\u65b0\u5efa\u7f16\u53f7\u7248\u672c\uff0c\u65e7\u7248\u672c\u53ea\u9690\u85cf\u4e0d\u5220\u9664\uff1b\u53ef\u5237\u65b0\u9884\u89c8\u6216\u6309\u610f\u89c1\u4fee\u6539\u4ee5\u4fdd\u7559\u624b\u52a8\u8c03\u6574\u3002',
    'output': '\u5355\u56fe\u590d\u523b\u4ee5\u76ee\u6807\u539f\u56fe\u4e3a\u753b\u5e03\u548c\u5c3a\u5bf8\u4f9d\u636e\uff0c\u5e76\u7528\u540c\u673a\u4f4d\u771f\u5b9e\u4ea7\u54c1\u6295\u5f71\u7ea6\u675f\u7ed3\u6784\uff1b\u73af\u5883\u6e32\u67d3\u4f7f\u7528\u5f53\u524d\u76f8\u673a\u6216\u89c6\u53e3\u6355\u83b7\u753b\u9762\u3002\u6bcf\u6b21\u751f\u6210\u524d\u81ea\u52a8\u6da6\u8272\u3002',
}


def short_text(text, cells=32):
    text = str(text or '').replace('\n', ' ')
    out, used = '', 0
    for char in text:
        n = 2 if unicodedata.east_asian_width(char) in 'WF' else 1
        if used + n > max(8, cells) - 2:
            return out + '\u2026'
        out, used = out + char, used + n
    return out


def help_icon(layout, text='', *, key='', icon='QUESTION'):
    """Description() is the only place instructional copy is displayed."""
    message = text or HELP.get(key, '')
    if message:
        op = layout.operator('wondful.ui_help', text='', icon=icon, emboss=False)
        op.message = message


def field(layout, data, name, label='', **kwargs):
    row = layout.row(align=True)
    row.scale_y = INPUT_HEIGHT
    row.use_property_split = True
    row.use_property_decorate = False
    if label is None:
        row.prop(data, name, **kwargs)
    else:
        row.prop(data, name, text=label, **kwargs)
    return row


def text_input(layout, data, name, label, description='', *, rows=3):
    """Same field and expanded preview for every long text input, no extra nesting."""
    head = layout.row(align=True)
    head.label(text=label)
    help_icon(head, description)
    # A native multiline RNA field is not available in every supported Blender.
    box = layout.column(align=True)
    field(box, data, name)
    value = str(getattr(data, name, '') or '')
    if '\n' in value:
        for line in value.splitlines()[:rows]:
            box.label(text=short_text(line, 48))
    return box


def preview_scale(context):
    width = getattr(getattr(context, 'region', None), 'width', 360) or 360
    ui_scale = getattr(getattr(getattr(context, 'preferences', None), 'system', None), 'ui_scale', 1.0) or 1.0
    return max(3.0, min(PREVIEW_MAX, (width / max(0.5, ui_scale) - 70) / 20))


def preview_tile_count(image, fit='cover'):
    """Compatibility helper: each reference image uses ONE Blender icon.

    Adjacent native preview widgets always reserve their own button padding;
    even zero-gap rows cannot produce a seamless panorama. Do not split.
    """
    return 1


def preview(layout, context, image, fit='cover'):
    """Show one continuous photo preview without any native-widget seams.

    A single Blender icon has a square UI footprint. preview_cache builds a
    seam-free, aspect-preserving thumbnail over a softened image backdrop;
    the source remains completely visible and can be opened via Zoom.
    """
    import bpy
    row = layout.row(align=True)
    row.alignment = 'CENTER'
    if image is not None:
        from .preview_cache import icon_for_image
        icon = icon_for_image(image, fit='seamless')
        if not icon:
            try:
                icon = bpy.types.UILayout.icon(image)
            except (RuntimeError, ReferenceError, AttributeError):
                icon = 0
        if icon:
            row.template_icon(icon_value=icon, scale=preview_scale(context))
        else:
            row.label(text='', icon='IMAGE_DATA')
    else:
        row.scale_y = 2.0
        row.label(text='', icon='IMAGE_DATA')
    return row


def image_slot(layout, context, props, kind, *, item=None, index=0,
               title='', required=False, movable=False):
    """Identical title / preview / filename / controls / note for all uploads."""
    box = layout.box()
    header = box.row(align=True)
    header.label(text=title + (' *' if required else ''), icon='IMAGE_DATA')
    help_icon(header, key=kind)
    target = kind == 'TARGET'
    image = getattr(props, 'sb_reference_image', None) if target else getattr(item, 'image', None)
    preview(box, context, image, fit='contain' if kind == 'PRODUCT' else 'cover')
    meta = box.row(align=True)
    meta.label(text=short_text(getattr(image, 'name', '') or '\u672a\u9009\u62e9', 40))
    if image:
        action = meta.operator('wondful.preview_reference', text='', icon='ZOOM_ALL')
        action.image_name = image.name
        help_icon(meta, getattr(image, 'filepath', '') or getattr(image, 'name', ''))
    actions = box.row(align=True)
    actions.scale_y = ACTION_HEIGHT
    if target:
        actions.operator('wondful.target_load', text='\u66ff\u6362' if image else '\u9009\u62e9', icon='FILEBROWSER')
        actions.operator('wondful.target_paste', text='\u7c98\u8d34', icon='PASTEDOWN')
        remove = actions.row(align=True); remove.enabled = bool(props.sb_image_path)
        remove.operator('wondful.target_clear', text='', icon='X')
        field(box, props, 'sb_target_note', '\u53c2\u8003\u91cd\u70b9')
    else:
        op = actions.operator('wondful.replace_reference' if item else 'wondful.load_reference',
                              text='\u66ff\u6362' if item else '\u9009\u62e9', icon='FILEBROWSER')
        op.ref_kind = kind
        if item:
            op.item_index = index
        paste = actions.operator('wondful.paste_reference', text='\u7c98\u8d34', icon='PASTEDOWN')
        paste.ref_kind, paste.item_index = kind, index if item else -1
        remove = actions.row(align=True); remove.enabled = item is not None
        op = remove.operator('wondful.clear_reference', text='', icon='X')
        op.ref_kind, op.item_index = kind, index
        if item is not None:
            field(box, item, 'instruction', '\u53c2\u8003\u91cd\u70b9')
        else:
            empty_note = box.column(); empty_note.enabled = False
            field(empty_note, props, 'ui_empty_note', '\u53c2\u8003\u91cd\u70b9')
        if movable:
            moves = box.row(align=True)
            for direction, symbol, can_move in [('UP','TRIA_UP', index > 0), ('DOWN','TRIA_DOWN', index + 1 < len(getattr(props, {'STYLE':'style_images','PERSON':'person_images','PRODUCT':'product_images'}[kind])))]:
                part = moves.row(align=True); part.enabled = can_move
                op = part.operator('wondful.move_reference', text='', icon=symbol)
                op.ref_kind, op.direction, op.item_index = kind, direction, index
    return box


def path_preview(layout, context, raw, title, description=''):
    """Advanced file inputs use the same one-piece preview as main images.

    Path loading is deferred to preview_cache's timer; Panel.draw does not
    read files or split thumbnails into independent Blender icon widgets.
    """
    from .preview_cache import icon_for_path
    box = layout.box()
    head = box.row(align=True)
    head.label(text=title, icon='IMAGE_DATA')
    help_icon(head, description or str(raw or ''))
    icon = icon_for_path(raw, fit='seamless')
    row = box.row(align=True)
    row.alignment = 'CENTER'
    if icon:
        row.template_icon(icon_value=icon, scale=preview_scale(context))
    else:
        row.scale_y = 2.0
        row.label(text='', icon='IMAGE_DATA')
    name = str(raw or '').replace(chr(92), '/').rsplit('/', 1)[-1]
    foot = box.row(align=True)
    foot.label(text=short_text(name or '未选择', 38))
    if icon:
        action = foot.operator('wondful.preview_reference', text='', icon='ZOOM_ALL')
        action.image_path = str(raw)
    return box


def path_slot(layout, context, data, prop, title, description=''):
    """Advanced file pickers share the main cards' title, preview and filename."""
    box = path_preview(layout, context, getattr(data, prop, ''), title, description)
    field(box, data, prop, '')
    return box
