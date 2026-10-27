# Wondful AI Renderer · 图片预览黑缝修复（UI-r5）

## 来源和症状

用户在 Blender 侧栏发现产品身份参考、主环境参考中间各有一条明显的竖向黑线。
此前 UI-r4 将横向参考图拆成两块独立的方形 `template_icon`，三视图甚至拆为三块；
Blender `UILayout` 为每个图标保留独立控件内边距，即使 `row(align=True)` 也不能保证像素级紧贴。
该黑线属于 Blender UI 控件之间的缝隙，不存在于原始图片文件中。

## r5 修复

- `ui_components.preview`：无论产品 / 环境 / 目标 / 人物图片的宽高比如何，**仅调用一次** `template_icon`。
- `ui_components.path_preview`：高级文件路径图片也**只画一次**预览图标，不再并排构建多张 tile。
- `preview_cache._pixels_thumbnail`：新增 `seamless` 生成策略。
  将**整张图**等比缩放放在方形缩略图中心，周围用原图居中取样并模糊的背景填充，
  轻微柔化前景边缘，避免黑色上下留白；车头、车尾不会被边缘裁掉。
- 移除 `tile:i:n` 预览模式的调用，显式拒绝新建 tile 以防拼接问题回归。
- 后台定时缓存保留，**没有在 `Panel.draw()` 中增加磁盘 IO 或图像解码**。
- `wondful_ai_renderer.__init__.BUILD_ID` 更新为 `two-workflows-ui-r5-seamless`；注册版本仍为 `1.2.4`。

## 设计取舍

Blender 原生 `UILayout.template_icon` 是**正方形图标控件**，无法在普通侧栏获得任意宽高比、无边框的整幅矩形图片。
本轮优先修复用户指出的**中间竖线**，并保证整图、车辆两端和产品三视图完整可见。
为了避免回到纯黑大边框，单张图标的方形空余部分采用柔和原图背景，而不是纯黑。
代价是宽图的中心清晰区域在竖向上只占方形图标的一部分，需要看细节时点击放大查看原图。
未使用 GPU 叠加绘制、魔改 Blender UI 或图像内容拉伸的方案，以免破坏侧栏兼容性。

## 验证

- `python -m unittest discover -s tests -q`：**156 项通过 / 0 失败**（r5 交付源码离线环境）。
- UI 模拟检查：所有主图片卡片和高级图片路径输入仅生成**一个** `template_icon`。
- 图片像素检查：16:9、4:1 超宽、1:4 竖图，源图完整、缩略图中线无黑分隔、源图像素不修改。
- `python -m compileall`：通过。
- Blender 安装 ZIP 和完整源码 ZIP 解压校验、文件哈希校验通过。
- **未进行 Blender 原生 UI 像素截图验收**：当时运行环境未安装 Blender；无法验证不同系统和 DPI 下的实际尺寸。

## Blender 实机检查

1. 从磁盘安装 r5 并重启 Blender。确认顶部悬停显示 `two-workflows-ui-r5-seamless`。
2. 重新打开用户截图中使用的原车图、环境图，预览区应各有**单张连续图像**，无中间竖线。
3. 测试 16:9、三视图、竖图、打包到 `.blend` 的图片和高级路径图片。
4. 点击放大，核对原图未被更改；如果看到旧两块图标，请先确认没有启用旧插件副本并重新启动 Blender。

## GitHub 状态

r5 代码已作为增量提交加入 `main`，保留此前修复历史；尚未发布正式插件 Release。
