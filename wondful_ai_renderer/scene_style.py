"""1.2.6 风格与光线锁定: the reference look (画风 + night lighting) read from the ④ 场景粗模 analysis / the environment
description / the user's 一句话需求, for the 主视角 prompt, 多角度 new views and the 场景补全 block hints.

Real tests 2026-10-20: the cyber night street lost its neon in the new views, and a Minecraft-style village came out
photoreal unless the user typed 「我的世界方块风格」 — the simple-path prompt hard-coded 「写实的商业广告摄影照片」
「照片质感真实自然」 and the analysis' "voxel-built … pixelated material variation" never reached the image model.
Realistic scenes keep their wording unchanged. Pure Python (no bpy)."""
from __future__ import annotations

# keyword → Chinese style word (English from the analysis / environment description, Chinese from the brief)
STYLE_WORDS = (
    ("minecraft", "我的世界式方块"), ("我的世界", "我的世界式方块"), ("voxel", "体素方块"), ("体素", "体素方块"),
    ("block-built", "方块搭建"), ("方块", "方块"),
    ("pixelated", "像素纹理"), ("pixel art", "像素画"), ("pixel-art", "像素画"), ("像素风", "像素风"),
    ("像素纹理", "像素纹理"), ("像素画", "像素画"),
    ("low-poly", "低多边形"), ("low poly", "低多边形"), ("低多边形", "低多边形"),
    ("cel-shad", "赛璐璐着色"), ("anime style", "动漫"), ("anime-style", "动漫"), ("动漫", "动漫"), ("二次元", "二次元"),
    ("cartoon", "卡通"), ("卡通", "卡通"),
    ("插画风", "插画"), ("手绘风", "手绘"), ("watercolor style", "水彩"), ("watercolour style", "水彩"), ("水彩风", "水彩"),
    ("油画风", "油画"), ("claymation", "黏土动画"), ("黏土风", "黏土质感"), ("风格化", "风格化"),
)
# phrases that must not count ("not stylized", "photorealistic", the massing text's 「纯色方块」, square floor tiles)
NEGATED = ("not cartoon", "non-cartoon", "photorealistic", "photo-realistic", "不是卡通", "非卡通", "纯色方块", "方块地砖",
           "方块砖", "小方块灯")
NIGHT_WORDS = ("night", "neon", "夜", "霓虹")
SIGN_WORDS = (("neon", "霓虹灯"), ("lightbox", "灯箱"), ("light box", "灯箱"), ("lantern", "灯笼"),
              ("billboard", "广告牌"), ("advertis", "广告牌"), ("sign", "招牌"), ("shop window", "橱窗"),
              ("street lamp", "路灯"), ("streetlight", "路灯"), ("street light", "路灯"),
              ("霓虹", "霓虹灯"), ("灯箱", "灯箱"), ("灯笼", "灯笼"), ("招牌", "招牌"), ("广告", "广告牌"), ("路灯", "路灯"))


def analysis_text(analysis) -> str:
    """The descriptive words of a ④ analysis (summary, description, Chinese lines, sky / ground words)."""
    an = analysis or {}
    pal = an.get("palette") or {}
    return " ".join(str(x or "") for x in (an.get("scene_summary"), an.get("environment_description"),
                                           an.get("environment_zh"), pal.get("lock_zh"), pal.get("sky_words"),
                                           pal.get("ground_words")))


def art_style(*texts) -> list:
    """Chinese style words of a non-photoreal look found in ``texts`` (ordered, unique); [] = realistic."""
    t = " ".join(str(x or "") for x in texts).lower()
    for neg in NEGATED:
        t = t.replace(neg, " ")
    out = []
    for key, zh in STYLE_WORDS:
        if key in t and zh not in out and not any(zh in o or o in zh for o in out):
            out.append(zh)
    return out[:4]


def style_phrase(words) -> str:
    return "、".join(words or [])


def is_night(*texts) -> bool:
    t = " ".join(str(x or "") for x in texts).lower()
    return any(w in t for w in NIGHT_WORDS)


def sign_words(*texts) -> list:
    t = " ".join(str(x or "") for x in texts).lower()
    out = []
    for en, zh in SIGN_WORDS:
        if en in t and zh not in out:
            out.append(zh)
    return out[:5]


def master_lock(analysis, brief="") -> str:
    """风格与光线锁定 for the 主视角 prompt from the ④ analysis ("" for a realistic day scene: its prompt stays as it
    was). Stylized → the 画风 must-keep line; night → the light-source / sign lock."""
    text = analysis_text(analysis)
    style = art_style(text, brief)
    parts = []
    if style:
        parts.append(f"画风锁定（来自环境参考图分析，必须保持）：整个环境是{style_phrase(style)}风格，"
                     "建筑、地面、植物和道具都保持参考图的造型语言和材质表现（方块 / 像素 / 笔触等同参考），"
                     "不要画成写实照片或普通写实建筑。")
    if is_night(text):
        signs = style_phrase(sign_words(text)) or "招牌、灯箱、橱窗和路灯"
        parts.append(f"光线锁定（来自环境参考图分析，必须保持）：夜景，天空保持夜色，人工光为主；街边建筑上布满真的在发光的{signs}，"
                     "保留局部亮光、光晕和强烈明暗对比；不要画成白天、黄昏或阴天，不要整体压成单一棕灰色调。")
    return "".join(parts)


def hint_suffix(analysis) -> str:
    """Appended to the generic 场景补全 block hints (电杆、推测建筑…) so they carry the scene's look."""
    text = analysis_text(analysis)
    style = art_style(text)
    out = []
    if style:
        out.append(f"画风：{style_phrase(style)}，与主视角一致")
    if is_night(text):
        out.append("夜景：与主视角同样亮着的招牌和灯光")
    return ("（" + "；".join(out) + "）") if out else ""
