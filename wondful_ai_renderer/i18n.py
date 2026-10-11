"""English translations for the add-on UI, registered through bpy.app.translations.

The source strings stay in Chinese. Blender shows the ``en_US`` entries when the
interface language is English. ``zh_HANS`` has no entries, so Chinese remains the
default there.

Blender keys a translation by ``(context, msgid)``:
- ``"*"`` covers ordinary UI text (labels, buttons, property names) and tooltips.
- ``"Operator"`` covers operator names (``bl_label``), which Blender also shows in tooltips.

Text that is built at draw time from several parts cannot match a key as a whole,
so it is translated explicitly with ``bpy.app.translations.pgettext_iface``. Its msgid
must match the literal in the UI code exactly.
"""

DEFAULT = "*"
OPERATOR = "Operator"

TRANSLATIONS = {
    "en_US": {
        # ① Connect step (wondful_ai_renderer/ui.py)
        (DEFAULT, "连接"): "Connect",
        (DEFAULT, "未检测"): "Not checked",
        (DEFAULT, "找不到 {provider}"): "Cannot find {provider}",
        (DEFAULT, "未登录"): "Signed out",
        (DEFAULT, "异常"): "Error",
        (DEFAULT, "正在连接"): "Connecting",
        (DEFAULT, "正在连接 AGY"): "Connecting to AGY",
        (DEFAULT, "使用 ChatGPT 登录"): "Sign in with ChatGPT",
        (DEFAULT, "Google 登录"): "Sign in with Google",
        (DEFAULT, "验证登录"): "Verify sign-in",
        (DEFAULT, "终端和浏览器里登录完，回来点验证登录。"): "Finish signing in in the terminal or browser, then click Verify sign-in.",
        (DEFAULT, "已装 Codex 仍找不到？在高级 › 插件偏好设置里填路径。"): "Codex is installed but still not found? Enter its path under Advanced › Add-on preferences.",
        (DEFAULT, "等待浏览器授权"): "Waiting for browser authorization",
        (DEFAULT, "打开本次登录页"): "Open this sign-in page",
        (DEFAULT, "授权码"): "Authorization code",
        (DEFAULT, "提交授权码"): "Submit code",
        (DEFAULT, "取消连接"): "Cancel connection",
        # Operator names, shown in the tooltips of the buttons above.
        (OPERATOR, "检测 Codex"): "Check Codex",
        (OPERATOR, "检测 AGY 安装"): "Check AGY installation",
        (OPERATOR, "使用 ChatGPT 登录"): "Sign in with ChatGPT",
        (OPERATOR, "验证登录"): "Verify sign-in",
        (OPERATOR, "Google 登录"): "Sign in with Google",
        (OPERATOR, "打开本次登录页"): "Open this sign-in page",
        (OPERATOR, "提交授权码"): "Submit code",
        (OPERATOR, "取消连接"): "Cancel connection",
        # Operator descriptions (tooltip body).
        (DEFAULT, "检测 Codex CLI 与 ChatGPT OAuth 登录状态"): "Check the Codex CLI and ChatGPT OAuth sign-in status",
        (DEFAULT, "调用官方 Codex OAuth 登录；通常会自动打开浏览器"): "Sign in with the official Codex OAuth flow; usually opens your browser automatically",
        (DEFAULT, "仅检测 CLI 路径与版本，不发起模型请求或浏览器登录"): "Check only the CLI path and version; makes no model request and no browser sign-in",
        (DEFAULT, "使用选定模型检查官方 CLI 登录；发起一次最小文本请求，不生成图片"): "Check official CLI sign-in with the selected model; sends one minimal text request and generates no images",
        (DEFAULT, "打开官方 AGY 交互终端，完成首次设置和 Google 登录；完成后回插件点验证登录"): "Open the official AGY terminal to finish first-time setup and Google sign-in; then come back here and click Verify sign-in",
    },
}
