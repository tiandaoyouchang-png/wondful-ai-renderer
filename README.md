# Wondful AI Renderer — Open-Source Blender AI Rendering Add-on

**中文：开源 Blender AI 渲染插件 · 产品可视化与汽车场景图**

> Create AI-assisted commercial-style product scenes from a real Blender 3D model, a fixed camera, and environment reference images. Blender anchors the composition; AI develops the environment and lighting; silhouette checks and detail restoration help preserve product shape, scale, colour, and placement.
>
> 使用真实 Blender 3D 模型、固定机位与环境参考图生成商业级产品/汽车场景图。Blender 锚定构图，AI 生成环境与光照，再通过轮廓验收和细节回贴，尽量保留产品造型、比例、颜色与位置。

| 跑鞋 · 城市雨夜 | 腕表 · 黑色大理石 | 单椅 · 北欧客厅 |
|---|---|---|
| ![](docs/showcase/shoe_rain.jpg) | ![](docs/showcase/watch_marble.jpg) | ![](docs/showcase/chair_nordic.jpg) |
| **台灯 · 咖啡馆** | **复古相机 · 街头胶片** | **小米 YU7 · 草地** |
| ![](docs/showcase/lamp_cafe.jpg) | ![](docs/showcase/camera_street.jpg) | ![](docs/showcase/yu7_grass.jpg) |

**适合谁**：电商卖家做主图和场景图、工业设计师出方案效果图、品牌做一款产品多种场景的广告图、3D 艺术家省掉搭场景和打灯的时间。

**付费版**：[Gumroad · $29.90](https://tiandao1.gumroad.com/l/wondful)（含 1 年更新）。开源版和付费版功能相同；付费版是打包好的安装包和 1 年版本更新，也是对项目的支持。

Wondful AI Renderer 是一个开源的 Blender 插件。一句话：**AI 负责整体商业视觉，Blender 负责产品真实性。** 插件从你的相机渲出白模底图，配上产品三视图和读成文字的环境参考，交给已登录的 ChatGPT（Codex）直接在白模上做图像编辑；出图后再把 Blender 里的真实细节（车牌、Logo、纹理）贴回，并按材质把颜色拉回模型。

English: an open-source Blender add-on for **shape-locked AI product rendering**. Shoes, watches, lamps, cameras, furniture, cars — any product with a 3D model. One camera setup becomes many photoreal commercial scenes, while silhouette, scale, colour and position stay locked to your Blender render. Blender locks the composition, the three-view sheet locks identity, the AI paints light and environment, and Blender then pastes real details and per-material colour back. The open-source and paid builds have the same features; the [Gumroad](https://tiandao1.gumroad.com/l/wondful) version is the packaged installer with a year of updates.

---

## 效果展示

每个案例的**完整提示词、参考图、插件参数和逐张轮廓数据**见 [docs/CASES.md](docs/CASES.md)（English: [docs/CASES.en.md](docs/CASES.en.md)）。数据如实标注，未达标的场景也写明了原因。

### 球鞋 · 腕表 · 复古相机 · 台灯

| 跑鞋 · 运动场晨光 | 跑鞋 · 城市雨夜 | 腕表 · 黑色大理石 |
|---|---|---|
| ![](docs/showcase/shoe_track.jpg) | ![](docs/showcase/shoe_rain.jpg) | ![](docs/showcase/watch_marble.jpg) |
| **腕表 · 岩石登山** | **台灯 · 咖啡馆** | **台灯 · 极简卧室** |
| ![](docs/showcase/watch_rock.jpg) | ![](docs/showcase/lamp_cafe.jpg) | ![](docs/showcase/lamp_bedroom.jpg) |
| **复古相机 · 街头胶片** | **复古相机 · 复古书房** | |
| ![](docs/showcase/camera_street.jpg) | ![](docs/showcase/camera_study.jpg) | |

模型均来自 [Khronos glTF Sample Assets](https://github.com/KhronosGroup/glTF-Sample-Assets)：MaterialsVariantsShoe（© 2021 Shopify, CC BY 4.0）、ChronographWatch（© 2025 Darmstadt Graphics Group, CC BY 4.0，表盘 Logo 为商标）、AntiqueCamera（© 2018 UX3D, CC0）、IridescenceLamp（© 2022 Wayfair, CC BY 4.0）。参考图均为 AI 生成。

### 家具：设计单椅 · 四种家居风格

| Blender 白模 | Blender 底图 | 北欧客厅 |
|---|---|---|
| ![](docs/showcase/chair_clay.jpg) | ![](docs/showcase/chair_base.jpg) | ![](docs/showcase/chair_nordic.jpg) |
| **侘寂茶室** | **工业 Loft** | **地中海露台** |
| ![](docs/showcase/chair_wabi.jpg) | ![](docs/showcase/chair_loft.jpg) | ![](docs/showcase/chair_terrace.jpg) |

| 场景 | 轮廓 IoU | 外轮廓 ≤3 px | 底部偏差 |
|---|---|---|---|
| 北欧客厅 | 99.2% | 100% | 0 px |
| 侘寂茶室 | 97.9% | 92% | 0 px |
| 工业 Loft | 99.0% | 99% | +1 px |
| 地中海露台 | 98.4% | 99% | 0 px |

### 汽车：小米 YU7 · 一个底图，四种户外场景

| Blender 白模 | 雪坡 | 草地 |
|---|---|---|
| ![](docs/showcase/yu7_clay.jpg) | ![](docs/showcase/yu7_snow.jpg) | ![](docs/showcase/yu7_grass.jpg) |
| **沙漠** | **泥地** | |
| ![](docs/showcase/yu7_sand.jpg) | ![](docs/showcase/yu7_mud.jpg) | |

同一个 3/4 前侧低机位。以 Blender 底图为基准做轮廓检测：

| 场景 | 轮廓 IoU | 外轮廓 ≤3 px | 底部偏差 |
|---|---|---|---|
| 雪坡 | 96.8% | 84% | 0 px |
| 草地 | 97.2% | 88% | +2 px |
| 沙漠 | 96.7% | 86% | +4 px |
| 泥地 | 96.2% | 86% | +1 px |

> 指标用 GrabCut 分割生成图中的产品，再与 Blender 渲染的产品 Mask 对比，属于近似测量。

---

## 核心能力

- **白模相机即构图权威**：输出与 Blender Camera 完全同机位；比例不一致时等比缩放、透明留边，不裁切、不拉伸。
- **白模底图 + 三视图**：生图主链路只发带地面和灯光的白模底图（锁构图）和产品三视图（锁身份），环境参考先读成文字。Depth / Normal / 部件 ID 仍会渲出，用于验收、分层和细节回贴，不再作为主要生图输入。
- **Identity Preserve Mask**：按部件锁定产品身份区域（Logo、表盘、灯组、鞋标、车标等），只放开环境、光影与材质质感。
- **严格模式验收 + 纠偏**：每张候选图按比例、位置、轮廓检查，不合格自动用 Mask 局部修复重试。
- **环境图读成文字，构图零漂移**（3.2.2）：只把带地面灯光的白模底图发给图像编辑模型，环境参考图先读成精准文字描述再参与生成，产品位置、大小、轮廓不再漂移；支持斜坡和改视角。
- **坡顶地面**（3.2.3）：高级 › 场景 › 地面（简洁输入下）可选「地面：自动 / 坡顶」，坡顶会自动铺一块前后向下倾斜的临时地形（下坡角度可调），适合山顶、观景台、坡顶陈列的画面；场景里已有自己的地形时始终用你的。
- **1.0 正式版 · 广告级质感**：环境参考图如果是专业广告图（如汽车广告），插件会额外读出「表面质感」（清漆光泽、天空和山体倒影、顶部高光带、暗部过渡），只把这套质感用到你的产品上，造型、比例和颜色仍以白模为准。这是第一个正式版，之前的 3.x 均为测试版。
- **山头 + 车漆质感**（3.2.4）：「坡顶」升级为「山头」——车停在圆顶小山最高处，四周地面下落（默认 18°，5–35° 可调），配合低机位仰拍；白模底图改用户外天空渐变 + 两条柔光灯带，车漆出生图更有清漆分量感、倒影和高光带。
- **自定义地形**（1.1.0）：高级 › 场景 › 地面 → 自定义，左前 / 右前 / 左后 / 右后四个角各设地面坡度（-35°…+35°），加「坡顶」和「凹凸」强度；预设：坡顶、对角扭曲、镜像扭曲、颠簸路面、爬坡、下坡、侧坡。产品按刚体临时落在地面上（三点着地、扭曲地面上一角略微悬空，渲染后自动恢复），提示词同步写入地形描述，车和非车产品都适用。
- **分层输出**（1.2.0）：高级 › 输出 › 分层输出，每张结果另存主体透明 PNG、投影层、干净背景（「完整」模式）和分层 PSD，方便在 PS / 剪辑里换背景、调投影。
- **环境体块 + 建筑参考**（1.2.0）：在 Blender 里用简单体块摆出产品周围的建筑（放进「环境体块」集合），AI 按场景风格画成真实建筑，产品保持锁定；每个体块可挂 1–3 张自己的建筑参考图；也可用正立面 / 侧立面 / 平面三视图直接雕出建筑体块。体块保持位置、高度、轮廓和透视。
- **细节回贴**（1.2.0）：生成后把 Blender 里的真实细节贴回成片——车牌、Logo 等标为「原样保留」的部件原样贴回，哑光部件纹理回贴，其余按频率融合；可选「按材质校色」（默认强度 0.7），把每种材质的颜色拉回模型、保留环境色调。
- **固定坐标轮廓验收 + 有界重生成**（1.2.3，未发布）：每次生成后按 Blender 产品遮罩在固定坐标上验收 IoU ≥ 0.95，不过线从原始白模重画，每张最多 3 次；额度用完或登录失效立即停止，不再改用代理生成。
- **多角度场景一致**（1.2.3，高级 › 多角度）：一个相机渲染满意后设为主视角，再选中其他相机一键渲染：已完成视角的画面投影到环境体块上作为新相机的底图，光照方向在世界里固定，离主视角最近的先渲染；每个相机 1 次生图，产品超标或环境明显不一致时最多重试 1 次。
- **自动接续生图 + 抠图 / 回贴修复**（1.2.2）：「生成」写好提示词后自动开始 AI 渲染，失败时面板显示原因；分层输出主体不再长进草地泥浆，细节回贴与按材质校色避开泥浆、喷溅和阴影，镜像车牌 / 标识会提示。
- **产品还原度**（1.2.1）：每次生成后在「高级 › 诊断 › 上次渲染」用像素给出偏移、缩放、轮廓误差、细节边缘误差和颜色 ΔE，并点名偏差大的部件。
- **在白模上直接编辑**（3.2.1）：Codex 默认直连托管图像编辑（gpt-image-2，复用 Codex 的 ChatGPT 登录），以白模底图为画布只改材质、灯光和环境；修复轮次把修复范围作为编辑 Mask 发送。失败时自动回退到 Codex CLI 代理生成。
- **双 Provider**：⑥ 选择每次生成 1–4 张（Codex 最多 4 张，Antigravity 最多 2 张）；推理模型列表合并账号实时列表与本机官方 CLI（1.2.2），本机 CLI 太旧不认新模型时自动改用较旧模型并在面板提示更新 CLI。
- **参考图管理**：产品参考决定身份造型，环境参考决定受光、反射和氛围；超出容量的参考自动整理成带编号的图集。
- **AI 润色提示词**：给参考图即可，AI 先分析环境的主光、色温和氛围（结果可编辑），再写成完整提示词；也可输入中文创意需求一键扩写。
- **产品外观锁定**（3.1.7）：配色与材质默认读取 Blender 材质；锁定的产品质感与“必须保留的细节”原样进入每个场景，换环境只重写环境与光影。
- **Compare 对比视图**：底图与结果叠加对比，所有候选图写入输出目录。

## 工作原理

```text
Blender 场景 + 相机 + 产品集合
        │
        ▼
 白模底图（锁构图） + 产品三视图（锁身份）
        │
        ▼
 环境参考 → 读成文字描述 → 写入提示词
        │
        ▼
 Codex 图像编辑直连（白模为画布；失败回退 Codex CLI / Antigravity CLI）
        │
        ▼
 比例适配 → 轮廓验收
        │
        ▼
 细节回贴（车牌 / Logo 原样贴回，纹理高频融合） → 按材质校色
        │
        ▼
 输出目录（可选分层 PNG / PSD） + Blender 内预览 / Compare
```

## 环境要求

- Blender **4.3 及以上**
- 本机安装并登录以下任一官方 CLI：
  - [Codex CLI](https://github.com/openai/codex)（需要具备图像生成能力的账号）
  - Antigravity CLI（Google 账号登录）
- 可选：`TYPESAFE_API_KEY`（Jev 语义引擎，无 Key 时自动使用本地规则）

## 安装

1. 下载仓库，将 `wondful_ai_renderer/` 文件夹打成 ZIP（或使用 Releases 中的安装包）。
2. Blender → 编辑 → 偏好设置 → 插件 → 从磁盘安装 → 选择 ZIP → 启用。
3. 在 3D 视图侧边栏（N 面板）找到 **Wondful AI 渲染器**，确认面板显示的版本号。

## 快速上手

1. 把产品放进一个集合，设置好相机（建议 3/4 前侧、略低机位）。
2. 按面板六张卡片走（完成的步骤编号会变成 ✓，还没轮到的卡片显示为灰色，但按钮照样能点）：
   - ① 连接：Codex 点「使用 ChatGPT 登录」；Antigravity 点「Google 登录」后再点「验证登录」。已连接时只剩一行“ChatGPT 已连接”。
   - ② 产品三视图：放一张产品三视图（正/侧/后拼成一张），面板直接显示大缩略图；下方选产品集合。
   - ③ 环境参考图：第一张是主光来源（大图），其余为辅助小图，可调顺序。
   - ④ 场景粗模（可选）：用 ③ 的第一张参考图（或另选一张）让 AI 搭出地形 / 道路 / 建筑 / 树的灰模，可同时设置相机，约 3–5 分钟；进度和「取消」在这张卡片里，⑥ 也显示一行状态。不需要就跳过。
   - ⑤ 提示词：可写一句需求，例如“雪地清晨，冷色调”；需要时展开「查看 / 编辑完整提示词」。留空则生成时自动写。
   - ⑥ 生成：点大按钮「生成」。生成中显示 读产品图 → 分析环境 → 写提示词 → 渲染验收 四个阶段和进度条，可随时「取消」。
3. 结果出现在 ⑥ 卡片里，只有三个按钮：「对比」「再来一张」「打开文件夹」。
4. 想手动调整时展开「高级」，按分组切换：模型（AI 服务、推理模型、登录）· 输出（流程、模式、输出目录、分层输出）· 产品（轮廓验收、细节回贴、外观锁定）· 场景（地面 / 地形、环境体块、环境分析、人物参考）· 多角度 · 诊断（上次渲染、复制诊断信息、偏好设置）都在这里；打开「分两步」可以先看提示词再渲染。

## 开发与测试

```bash
# 离线测试
python -m unittest discover -s tests -p "test_*.py" -v

# Blender 实机验收（无 GPU 可用 Cycles 代替 EEVEE 渲染结构图）
WONDFUL_STRUCTURE_ENGINE=CYCLES blender -b --factory-startup \
  --python tools/blender_acceptance.py -- --out /tmp/wondful_accept
```

## 仓库结构

```text
wondful_ai_renderer/       Blender 插件主体
skills/                    汽车渲染风格迁移技能包（可给 Codex 等 Agent 使用）
geometry-render-pipeline/  几何约束商业渲染管线 V0.1 原型
tests/                     离线测试
docs/showcase/             README 展示图
```

## 已知限制

- 生图模型不是原生 ControlNet，无法保证像素级对齐，偶尔会把产品略放大或挪位；GPT Image 的编辑 Mask 是提示性的（模型参考 Mask，但不保证 Mask 外逐像素不变）。严格模式的验收与 Mask 纠偏用来兜底。
- Codex 图像编辑直连使用的是 Codex 自己的 ChatGPT 后端接口（非公开 API），接口变化时会自动回退到 CLI 代理生成；需要 Codex 使用文件方式保存登录（`auth.json`）。
- 实际出图质量取决于 CLI 背后的图像模型与账号额度；严格模式的远程调用次数可能多于最终导出张数。
- Blender 5.x 的合成器 API 有较大变化，自 3.1.6 起已适配（实测 Blender 5.1.2）。

## 致谢

- 小米 YU7 模型：Sketchfab，作者 Ddiaz Design（展示用途）
- 单椅模型：Khronos glTF Sample Assets · SheenChair，© 2020 Wayfair LLC，CC0 1.0

## 参与贡献

欢迎提 Issue 和 PR：新场景预设、更多 Provider、更好的验收算法、其他品类的案例都非常欢迎。觉得有用请点个 ⭐ Star。

Contributing guide (English): [CONTRIBUTING.md](CONTRIBUTING.md)

## License

[MIT](LICENSE)
