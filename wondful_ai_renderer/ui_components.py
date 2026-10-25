"""Shared, quiet panel primitives. No rendering/network/file IO during draw."""
import unicodedata

INPUT_HEIGHT = 1.2
ACTION_HEIGHT = 1.35
PREVIEW_MAX = 10.0

HELP = {
    'environment': '\u4f7f\u7528\u5f53\u524d\u76f8\u673a\u548c\u771f\u5b9e\u8f66\u6a21\uff0c\u73af\u5883\u56fe\u4ec5\u63d0\u4f9b\u6750\u8d28\u3001\u5149\u7167\u548c\u6c1b\u56f4\uff0c\u4e0d\u6539\u53d8\u673a\u4f4d\u3002',
    'reference': '\u5148\u751f\u6210\u76f8\u673a\u4e0e\u7c97\u6a21\uff0c\u518d\u9884\u89c8\u548c\u6309\u610f\u89c1\u4fee\u6539\u3002\u8f66\u578b\u59cb\u7ec8\u4ee5\u5bfc\u5165\u6a21\u578b\u4e3a\u51c6\u3002',
    'PRODUCT': '\u5fc5\u586b\uff1a\u9009\u62e9\u4e00\u5f20\u4ea7\u54c1\u8eab\u4efd\u53c2\u8003\uff0c\u53ef\u4f7f\u7528\u6b63\u3001\u4fa7\u3001\u540e\u4e09\u89c6\u56fe\u62fc\u56fe\u3002\u53ea\u53c2\u8003\u8f66\u578b\u7ec6\u8282\uff0c\u4e0d\u6cbf\u7528\u539f\u56fe\u6253\u5149\u3002\u6587\u4ef6\u65e0\u6548\u65f6\u4e0d\u53d1\u8d77\u5728\u7ebf\u8bf7\u6c42\u3002',
    'STYLE': '\u63d0\u4f9b\u73af\u5883\u5916\u89c2\u548c\u5149\u7167\uff0c\u4e0d\u63d0\u4f9b\u673a\u4f4d\u6216\u8f66\u578b\u3002\u53ef\u9009\u591a\u5f20\uff0c\u7b2c\u4e00\u5f20\u4e3a\u4e3b\u53c2\u8003\uff1b\u201c\u6dfb\u52a0\u201d\u4e0d\u4f1a\u8986\u76d6\u5df2\u6709\u56fe\u7247\u3002',
    'TARGET': '\u590d\u523b\u76ee\u6807\u56fe\uff0c\u4e0e\u73af\u5883\u53c2\u8003\u72ec\u7acb\u3002\u6362\u56fe\u540e\u539f\u7c97\u6a21\u4e0d\u4f1a\u88ab\u5220\u9664\uff0c\u4f46\u5fc5\u987b\u91cd\u65b0\u751f\u6210\u5339\u914d\u8be5\u76ee\u6807\u7684\u7c97\u6a21\u624d\u80fd\u7ee7\u7eed\u3002',
    'PERSON': '\u4eba\u7269\u5916\u89c2\u53c2\u8003\uff0c\u4e0d\u7528\u4e8e\u8f66\u578b\u8eab\u4efd\u6216\u76f8\u673a\u5339\u914d\u3002',
    'preview': '\u5237\u65b0\u4ec5\u6e32\u67d3\u5f53\u524d\u76f8\u673a\u4e0e\u7c97\u6a21\uff0c\u4e0d\u91cd\u65b0\u5206\u6790\u3002\u6309\u610f\u89c1\u4fee\u6539\u4f1a\u8c03\u7528\u5728\u7ebf\u5206\u6790\uff0c\u4ec5\u4fee\u6539\u63d2\u4ef6\u7c97\u6a21\u548c\u76f8\u673a\uff0c\u4e0d\u6539\u8f66\u8eab\u7f51\u683c\u6216\u6750\u8d28\u3002',
    'prompt': '\u5b8c\u6574\u63d0\u793a\u8bcd\u4f18\u5148\uff0c\u672a\u586b\u65f6\u4f7f\u7528\u4e00\u53e5\u8bdd\u9700\u6c42\u3002AI \u6da6\u8272\u53ef\u9009\uff0c\u4e24\u79cd\u6a21\u5f0f\u72ec\u7acb\u4fdd\u5b58\u8349\u7a3f\u3002',
    'build': '\u6839\u636e\u76ee\u6807\u56fe\u751f\u6210\u76f8\u673a\u548c\u7c97\u6a21\u3002\u518d\u6b21\u751f\u6210\u4f1a\u66ff\u6362\u63d2\u4ef6\u7c97\u6a21\uff1b\u4fdd\u7559\u624b\u52a8\u8c03\u6574\u8bf7\u4f7f\u7528\u5237\u65b0\u9884\u89c8\u6216\u6309\u610f\u89c1\u4fee\u6539\u3002',
    'output': '\u4f7f\u7528\u5f53\u524d\u76f8\u673a\u548c\u573a\u666f\u751f\u6210\u6210\u7247\uff1b\u751f\u6210\u524d\u81ea\u52a8\u68c0\u67e5\u5fc5\u586b\u53c2\u8003\u548c\u6765\u6e90\uff0c\u4e0d\u518d\u9700\u8981\u624b\u52a8\u52fe\u9009\u3002',
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
    return max(3.0, min(PREVIEW_MAX, (width / max(0.5, ui_scale) - 80) / 20))


def preview(layout, context, image):
    """One aspect-preserving native thumbnail size for every image slot."""
    import bpy
    row = layout.row(align=True)
    row.alignment = 'CENTER'
    icon = 0
    if image is not None:
        from .preview_cache import icon_for_image
        icon = icon_for_image(image)
        if not icon:
            try:
                icon = bpy.types.UILayout.icon(image)
            except (RuntimeError, ReferenceError, AttributeError):
                pass
    if icon:
        row.template_icon(icon_value=icon, scale=preview_scale(context))
    else:
        # Reserve the same visual well without pretending an image exists.
        row.scale_y = preview_scale(context)
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
    preview(box, context, image)
    meta = box.row(align=True)
    meta.label(text=short_text(getattr(image, 'name', '') or '\u672a\u9009\u62e9', 40))
    if image:
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
    """Shared read-only image well; thumbnails load in the deferred cache."""
    from .preview_cache import icon_for_path
    box = layout.box()
    head = box.row(align=True)
    head.label(text=title, icon='IMAGE_DATA')
    help_icon(head, description or str(raw or ''))
    icon = icon_for_path(raw)
    row = box.row(align=True)
    row.alignment = 'CENTER'
    if icon:
        row.template_icon(icon_value=icon, scale=preview_scale(context))
    else:
        row.scale_y = preview_scale(context)
        row.label(text='', icon='IMAGE_DATA')
    name = str(raw or '').replace(chr(92), '/').rsplit('/', 1)[-1]
    box.label(text=short_text(name or '\u672a\u9009\u62e9', 40))
    return box


def path_slot(layout, context, data, prop, title, description=''):
    """Advanced file pickers share the main cards' title, preview and filename."""
    box = path_preview(layout, context, getattr(data, prop, ''), title, description)
    field(box, data, prop, '')
    return box
