English version: [CASES.en.md](CASES.en.md)

# Wondful AI Renderer · 案例集（CASES）

同一个 Blender 机位渲出一张底图，换环境参考和环境提示词，批量出多场景。每张结果都和 Blender 渲染的产品 Mask 做轮廓比对。

## 验收方法与门槛

- **轮廓 IoU**：以 Blender Mask 为种子，用 OpenCV GrabCut 在生成图里分割产品，与 Mask 求交并比。门槛 **>95%**。
- **外轮廓 ≤3 px**：Blender Mask 外轮廓上的点，到生成图分割轮廓的距离 ≤3 px 的比例。门槛 **≥80%**。
- **底部偏差**：分割结果最低点与 Blender Mask 最低点的像素差（接地是否漂移）。门槛 **<5 px**（绝对值）。
- 生成图先缩放到底图分辨率（960×540）再比对。脚本：`cases/tools/sim.py`（与单椅案例的 `simc.py` 相同，单椅复测结果一致）。
- **补充指标（不计入门槛）**：Blender 外轮廓 3 px 内是否有生成图边缘（Canny），`cases/tools/edgechk.py`。它不依赖分割，用来区分“真的错位”和“GrabCut 分割失败”（透明玻璃、细脚架、深色背景、贴边道具）。纹理多的背景会让它偏宽松。
- 指标属于近似测量。

## 总览

| 案例 | 场景 | 轮廓 IoU | 外轮廓 ≤3 px | 底部偏差 | 补充边缘 | 结果 |
|---|---|---|---|---|---|---|
| 小米 YU7 | 雪坡 / 草地 / 沙漠 / 泥地 | 96.8% / 97.2% / 96.7% / 96.2% | 84% / 88% / 86% / 86% | 0 / +2 / +4 / +1 px | — | ✅ ×4 |
| SheenChair 单椅 | 北欧 / 侘寂 / Loft / 地中海 | 99.2% / 97.9% / 99.0% / 98.4% | 100% / 92% / 99% / 99% | 0 / 0 / +1 / 0 px | 99% / 100% / 98% / 100% | ✅ ×4 |
| 跑鞋 | 城市街头雨夜 | 95.9% | 79% | -4 px | 100% | ❌ |
| 跑鞋 | 运动场晨光 | 97.9% | 94% | +0 px | 99% | ✅ |
| 计时腕表 | 黑色大理石微距 | 97.8% | 84% | +0 px | 91% | ✅ |
| 计时腕表 | 户外岩石登山 | 96.4% | 70% | +0 px | 91% | ❌ |
| 复古相机 | 复古书房 | 93.7% | 91% | +3 px | 97% | ❌ |
| 复古相机 | 街头胶片风 | 90.4% | 84% | +2 px | 100% | ❌ |
| 幻彩台灯 | 极简卧室夜晚 | 66.8% | 47% | -199 px | 100% | ❌ |
| 幻彩台灯 | 咖啡馆 | 97.5% | 93% | -1 px | 94% | ✅ |

未达标的 5 张都在各自案例下写明了原因：叠图上产品位置基本没跑，主要是 GrabCut 在透明玻璃、细三脚架、深色背景和贴边道具上分割失败。复古相机已按真实材质（银色金属镜头、黑色皮革机身）重做，两张都只差 IoU（外轮廓与底部达标）。

---

## 案例一：小米 YU7 · 一个底图，四种户外场景

| Blender 白模 | 雪坡 | 草地 |
|---|---|---|
| ![](showcase/yu7_clay.jpg) | ![](showcase/yu7_snow.jpg) | ![](showcase/yu7_grass.jpg) |
| **沙漠** | **泥地** | |
| ![](showcase/yu7_sand.jpg) | ![](showcase/yu7_mud.jpg) | |

- 模型：Sketchfab 免费模型，作者 Ddiaz Design（约 58 万面），仅作展示用途；仓库不分发模型文件。
- 机位：3/4 前侧低机位、略仰视，车停在坡顶 / 地面上；要求四个场景的地平线保持一致。
- 参考图：产品造型与漆面参考用的是小米 YU7 / YU7 GT 官方宣传图（版权归小米，仅作参考，不随仓库分发）；不放人物参考。
- 提示词要点（原始提示词未单独存档，以下为当时确定的要求）：
  - 产品外观：亮翡翠绿车漆（不压暗），厚清漆、大面积柔和长条渐变高光（参考 YU7 GT 官图）；必须保留贯穿灯带、五辐轮毂、后视镜、车窗线和“Xiaomi YU7”前牌照（第一版泥地出图时牌照被改掉，加了这条后修正）。
  - 环境/风格：雪坡冷亮、草地自然阳光、沙漠暖光扬沙、泥地阴天；地面颜色反到下车身，接地阴影要实，调色克制，细颗粒。
- 插件设置：当时未单独存档。

| 场景 | 轮廓 IoU | 外轮廓 ≤3 px | 底部偏差 |
|---|---|---|---|
| 雪坡 | 96.8% | 84% | 0 px |
| 草地 | 97.2% | 88% | +2 px |
| 沙漠 | 96.7% | 86% | +4 px |
| 泥地 | 96.2% | 86% | +1 px |

## 案例二：设计单椅 · 四种家居风格

| Blender 底图 | 北欧客厅 | 侘寂茶室 |
|---|---|---|
| ![](showcase/chair_base.jpg) | ![](showcase/chair_nordic.jpg) | ![](showcase/chair_wabi.jpg) |
| **工业 Loft** | **地中海露台** | |
| ![](showcase/chair_loft.jpg) | ![](showcase/chair_terrace.jpg) | |

- 模型：Khronos glTF Sample Assets · SheenChair，© 2020 Wayfair LLC，CC0 1.0（https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/SheenChair）。
- 底图：Blender 5.1.2，Cycles 48 spp，960×540，低机位 3/4 前侧，脚本 `chair_scene.py`；底图 `chair_base.png`，Mask `chair_mask.png`。
- 参考与提示词：每个场景一段环境/风格描述（北欧客厅 / 侘寂茶室 / 工业 Loft / 地中海露台）。原始提示词、环境参考和插件设置当时未单独存档，新案例已改为每个场景存一份 prompt.md。

| 场景 | 轮廓 IoU | 外轮廓 ≤3 px | 底部偏差 |
|---|---|---|---|
| 北欧客厅 | 99.2% | 100% | 0 px |
| 侘寂茶室 | 97.9% | 92% | 0 px |
| 工业 Loft | 99.0% | 99% | +1 px |
| 地中海露台 | 98.4% | 99% | 0 px |

## 案例三：跑鞋 · 城市雨夜 / 运动场晨光

| Blender 底图 | 白模 | 城市街头雨夜 | 运动场晨光 |
|---|---|---|---|
| ![](showcase/shoe_base.jpg) | ![](showcase/shoe_clay.jpg) | ![](showcase/shoe_rain.jpg) | ![](showcase/shoe_track.jpg) |

- 模型：Khronos glTF Sample Assets · MaterialsVariantsShoe（https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/MaterialsVariantsShoe）
- 授权与署名：© 2021 Shopify，CC BY 4.0（https://creativecommons.org/licenses/by/4.0/）。
- 底图：Blender 4.3.2，Cycles 64 spp，960×540，AZ=45 EL=1.6 TZ=0.4 FILL=0.75（相机在鞋头一侧 3/4 前侧，略俯视；50 mm）。使用 glTF 默认材质变体（浅蓝针织）。
- 设置：环境／风格组每个场景 1 张主参考，产品造型与人物组不放；渲染模式 标准；严格构图锁定 / 结构控制图 / 局部结构修复 开；Identity Preserve：关闭也可（鞋面无 Logo；中底的压印字样由底图保留）。
- 完整提示词、参考图、全部尝试和叠图：[`cases/shoe/prompt.md`](cases/shoe/prompt.md)

**产品外观（固定）**

```text
【产品外观】浅蓝色针织网面跑鞋，白色发泡中底，深灰色鞋带与鞋口内衬，鞋侧两块半透明灰蓝色 TPU 饰片，后跟提拉环。严格保持图1中鞋子的轮廓、比例、画面位置和视角完全不变，不改鞋型、鞋带走向和饰片形状，不添加 Logo 或文字。
```

**城市街头雨夜** · 环境参考 [`cases/shoe/refs/ref_rain.png`](cases/shoe/prompt.md)（见下图）（主参考；取：冷蓝环境光 + 左后方品红/青色霓虹轮廓光、湿地反光和雨丝氛围；不取：画面里的招牌与物体位置。）

<img src="showcase/refs/shoe_rain.jpg" width="360" alt="参考图">

```text
【环境/风格】城市街头雨夜：鞋子站在湿漉漉的黑色柏油路面上，远处积水倒映品红与青色霓虹和暖色路灯，背景大光斑虚化，空中细雨；冷蓝环境光，左后方霓虹轮廓光勾出鞋面边缘。鞋底正下方的路面偏暗，只有一条清晰的深色接触阴影，倒影很淡且不贴着中底，白色中底边缘与路面对比清晰。电影感球鞋广告大片。
English: light-blue knit running sneaker on a wet neon-lit city street at night in the rain; keep the exact silhouette, scale, position and camera angle of image 1.
```

**运动场晨光** · 环境参考 [`cases/shoe/refs/ref_track.png`](cases/shoe/prompt.md)（见下图）（主参考；取：右后方低角度金色晨光、长影、薄雾和红色跑道质感；不取：跑道上的物体位置。）

<img src="showcase/refs/shoe_track.jpg" width="360" alt="参考图">

```text
【环境/风格】运动场晨光：鞋子站在红色塑胶跑道上，白色分道线斜向延伸，右后方低角度金色晨光形成长影子和暖色轮廓光，轻薄晨雾，背景虚化的绿色草坪和看台，淡蓝天空；鞋底与跑道接触处有自然接触阴影。清新有活力的运动品牌广告。
English: light-blue knit running sneaker on a red running track at sunrise with golden rim light; keep the exact silhouette, scale, position and camera angle of image 1.
```

| 场景 | 轮廓 IoU | 外轮廓 ≤3 px | 底部偏差 | 补充边缘 | 结果 |
|---|---|---|---|---|---|
| 城市街头雨夜 | 95.9% | 79% | -4 px | 100% | ❌ |
| 运动场晨光 | 97.9% | 94% | +0 px | 99% | ✅ |

> 城市街头雨夜：未达标原因：外轮廓 ≤3 px 为 79%，差 1 个百分点。叠图上鞋形与 Blender 轮廓基本重合，偏差来自深色湿地面与深灰鞋带/后跟颜色接近，GrabCut 在鞋口附近分割不准。另：背景霓虹灯牌里有一个类似“%”的符号（非可读文字）。

> 运动场晨光：中底上的压印字样来自模型贴图，生成后保留。

## 案例四：计时腕表 · 黑色大理石微距 / 户外岩石登山

| Blender 底图 | 白模 | 黑色大理石微距 | 户外岩石登山 |
|---|---|---|---|
| ![](showcase/watch_base.jpg) | ![](showcase/watch_clay.jpg) | ![](showcase/watch_marble.jpg) | ![](showcase/watch_rock.jpg) |

- 模型：Khronos glTF Sample Assets · ChronographWatch（https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/ChronographWatch）
- 授权与署名：© 2025 Darmstadt Graphics Group GmbH，CC BY 4.0；模型与贴图 Eric Chadwick，源自 Sketchfab “Chronograph Watch Mudmaster”（graphiccompressor，CC BY 4.0）。表盘上的 Khronos / 3D Commerce / DGG 标志为各自商标，不在 CC BY 授权范围内。
- 底图：Blender 4.3.2，Cycles 64 spp，960×540，AZ=-25 EL=0.7 FILL=0.8（表盘朝前、3/4 左前方，略俯视；50 mm）。Blender 4.3.2 自带 glTF 导入器在该模型的 KHR_materials_variants 上报错（TypeError），所以先把 “Midnight Gold” 变体烘成默认材质、去掉 variants 扩展后再导入（几何不变）。
- 设置：环境／风格组每个场景 1 张主参考，产品造型与人物组不放；渲染模式 标准；严格构图锁定 / 结构控制图 / 局部结构修复 开；Identity Preserve：开启（表盘有字标/Logo），扩边 2 px；硬恢复关闭。
- 完整提示词、参考图、全部尝试和叠图：[`cases/watch/prompt.md`](cases/watch/prompt.md)

**产品外观（固定）**

```text
【产品外观】香槟金色金属表壳的多功能计时腕表，八角形表圈带银色螺丝和按键，黑色表盘、金色大号数字 12/3/6/9 与指针、下方液晶小窗，金色菱格纹表带与灰色塑料部件。严格保持图1中腕表的轮廓、比例、画面位置和视角完全不变，表盘指针位置与大号数字不变，不新增文字或 Logo。
```

**黑色大理石微距** · 环境参考 [`cases/watch/refs/ref_marble.png`](cases/watch/prompt.md)（见下图）（主参考；取：低调棚拍——左上窄长柔光条形成的长条高光、右侧暖金点缀光、深黑背景、抛光石面倒影；不取：石纹的具体走向。）

<img src="showcase/refs/watch_marble.jpg" width="360" alt="参考图">

```text
【环境/风格】黑色大理石微距：腕表站在抛光的黑色大理石台面上，石面有白色与淡金色细纹理，左上方一条窄长柔光箱在金属表壳和台面上形成长条高光，右侧微弱暖金色点缀光，背景沉入深黑，台面有腕表的清晰倒影。低调奢华腕表广告大片，微距质感。
English: champagne-gold chronograph watch standing on polished black marble, low-key luxury macro lighting; keep the exact silhouette, scale, position and camera angle of image 1.
```

**户外岩石登山** · 环境参考 [`cases/watch/refs/ref_rock.png`](cases/watch/prompt.md)（见下图）（主参考；取：左侧傍晚金色阳光、清晰硬阴影、花岗岩质感和远山虚化；不取：登山绳的位置（要求放在画面最右侧）。）

<img src="showcase/refs/watch_rock.jpg" width="360" alt="参考图">

```text
【环境/风格】户外岩石登山：腕表站在高山花岗岩岩脚上，岩面粗糙带地衣，远处蓝色山脊与山谷虚化；左侧傍晚金色阳光，清晰阴影，金属表壳反射暖光与天空蓝。橙色登山绳与钢扣只放在画面最右侧边缘，与腕表保持明显距离，不接触、不遮挡腕表；腕表底部与岩面之间有一条清晰的深色接触阴影。硬核户外工具表广告。
English: champagne-gold chronograph watch standing on a granite mountain ledge at golden hour; keep the exact silhouette, scale, position and camera angle of image 1.
```

| 场景 | 轮廓 IoU | 外轮廓 ≤3 px | 底部偏差 | 补充边缘 | 结果 |
|---|---|---|---|---|---|
| 黑色大理石微距 | 97.8% | 84% | +0 px | 91% | ✅ |
| 户外岩石登山 | 96.4% | 70% | +0 px | 91% | ❌ |

> 黑色大理石微距：表盘上的城市缩写、3D Commerce 字标等微小文字在 1376 px 输出中有轻微变形（底图本身也很小）；大号数字与指针保持正确。

> 户外岩石登山：未达标原因：IoU 与底部偏差达标，外轮廓 ≤3 px 只有 70%。叠图显示表壳本身与 Blender 轮廓重合，但两次生成都把钢扣/登山绳放在表壳右侧紧贴处（bbox 右侧 +18 px），GrabCut 把它并进了腕表。表盘微小文字有轻微变形。

## 案例五：复古相机 · 复古书房 / 街头胶片风

| Blender 底图 | 白模 | 复古书房 | 街头胶片风 |
|---|---|---|---|
| ![](showcase/camera_base.jpg) | ![](showcase/camera_clay.jpg) | ![](showcase/camera_study.jpg) | ![](showcase/camera_street.jpg) |

- 模型：Khronos glTF Sample Assets · AntiqueCamera（https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/AntiqueCamera）
- 授权与署名：© 2018 UX3D，CC0 1.0（https://creativecommons.org/publicdomain/zero/1.0/）；作者 Maximilian Kamps。模型上的 UX3D 标志为 UX3D 商标。
- 底图：Blender 4.3.2，Cycles 64 spp，960×540，AZ=-40 EL=0.65 TZ=0.5 FILL=0.86（3/4 左前方，站立视高略俯视；50 mm）。整机连三脚架一起作为产品。
- 设置：环境／风格组每个场景 1 张主参考，产品造型与人物组不放；渲染模式 标准；严格构图锁定 / 结构控制图 / 局部结构修复 开；Identity Preserve：开启（机身有 UX3D 标志），扩边 2 px；硬恢复关闭。
- **重做说明（2026-10-07）**：旧版产品外观误写为“木质机身、黄铜镜头”，模型实际是黑色皮革机身 + 银色镀铬镜头（只有三脚架是木质）。本版改正了产品外观（黄铜→银色），重新生成了两张参考图（街头参考不再有 “Caf” 字样）和两个场景；场景改为浅色墙面/地面衬托深色机身与木脚架，方便测量。每个场景本轮试了 2 次，全部尝试（第 1–4 次）在 `cases/camera/attempts/`。
- 完整提示词、参考图、全部尝试和叠图：[`cases/camera/prompt.md`](cases/camera/prompt.md)

**产品外观（固定）**

```text
【产品外观】复古折叠式风箱相机：机身为深灰黑色皮革包覆的箱体，黑色褶皱皮质风箱；镜头、镜头板、前组支架和镜头下方向右伸出的导轨平板为银色亮面金属（镀铬质感），镜头中心为深色玻璃；机身下方黑色金属云台（点缀银色小圆钉，左侧伸出黑色手柄），银色金属连接盘；安装在做旧的胡桃木三脚架上，每条腿是两根并排木条组成的宽扁双杆结构（侧面有浅色木纹），腿宽与图1完全一致、不要画细，脚末端为浅色金属包脚。严格保持图1中相机和三脚架的轮廓、比例、画面位置和视角完全不变，三条脚架的角度、宽度和落地点不变，镜头保持银色金属，不改成黄铜或金色，不添加文字或 Logo。
```

**复古书房** · 环境参考 [`cases/camera/refs/ref_study.png`](cases/camera/prompt.md)（见下图）（主参考；取：浅鼠尾草绿护墙板、大块米白地毯、左侧窗光 + 右侧黄铜台灯暖光、胶片颗粒；不取：书架与书桌的具体位置。）

<img src="showcase/refs/camera_study.jpg" width="360" alt="参考图">

```text
【环境/风格】复古书房：相机三条脚都稳稳站在一块大块浅米白色羊毛地毯的中央（地毯边缘离脚尖很远），地毯外露出浅橡木人字拼地板；相机正后方是从地脚到顶的浅鼠尾草绿色护墙板，深色机身和胡桃木脚架在浅绿背景前轮廓清晰；书架和旧皮面书（书脊无文字）只在画面左右两侧，右侧红木书桌上黄铜台灯发出暖黄钨丝光，左侧窗户投入柔和金色窗光；暖调复古、明亮通透，轻微胶片颗粒，银色镜头上有暖色反光，脚尖在地毯上有轻柔接触阴影。
English: antique black folding bellows camera with a silver chrome lens on a walnut wooden tripod, standing in the middle of a large cream rug in a bright 1920s study with pale sage-green panelled walls; keep the exact silhouette, scale, position and camera angle of image 1.
```

**街头胶片风** · 环境参考 [`cases/camera/refs/ref_street.png`](cases/camera/prompt.md)（见下图）（主参考；取：Portra 400 胶片色调与颗粒、粉彩薄荷绿灰泥墙、浅灰弹石路、左侧柔和暖光；不取：木门和自行车的位置。参考图无任何招牌或文字。）

<img src="showcase/refs/camera_street.jpg" width="360" alt="参考图">

```text
【环境/风格】街头胶片风：相机支在欧洲老城小巷浅灰白色的石灰岩弹石路上；相机正后方是虚化的粉彩淡薄荷绿色灰泥墙，深色机身和胡桃木脚架在浅绿背景前轮廓清晰；两侧虚化的绿色百叶窗、木门、米白色遮阳棚（无任何文字）和靠墙的自行车；左侧傍晚柔和暖光，阴影短而淡，脚尖在弹石上有轻柔接触阴影；Kodak Portra 400 胶片色调，细腻颗粒，高光微褪色。
English: antique black folding bellows camera with a silver chrome lens on a walnut wooden tripod on pale limestone cobblestones in an old European lane, pastel mint-green stucco wall behind, Portra 400 film look; keep the exact silhouette, scale, position and camera angle of image 1.
```

| 场景 | 轮廓 IoU | 外轮廓 ≤3 px | 底部偏差 | 补充边缘 | 结果 |
|---|---|---|---|---|---|
| 复古书房 | 93.7% | 91% | +3 px | 97% | ❌ |
| 街头胶片风 | 90.4% | 84% | +2 px | 100% | ❌ |

> 复古书房：未达标原因：IoU 93.7%（差 1.3 个点），外轮廓 ≤3 px 与底部偏差达标。漏分的是细部——云台左侧手柄、镜头下方向右伸出的导轨平板、右前脚架外侧的浅色木条边；机身、风箱、三条脚架和落地点与 Blender 轮廓基本重合。外观：镜头与支架为银色金属，但在暖色台灯光下镜头外圈有明显暖金色反光，近看偏香槟色，未完全达到底图的冷白镀铬。无可读文字。

> 街头胶片风：未达标原因：IoU 90.4%，外轮廓 ≤3 px 与底部偏差达标，bbox 四边偏差都在 1 px 内。右前脚架外侧的浅色木条被画窄/与浅绿墙面相融，云台手柄落在绿色百叶窗前被漏分。外观：镜头、支架、导轨为银色镀铬，与模型一致；无可读文字。

## 案例六：幻彩玻璃台灯 · 极简卧室夜晚 / 咖啡馆

| Blender 底图 | 白模 | 极简卧室夜晚 | 咖啡馆 |
|---|---|---|---|
| ![](showcase/lamp_base.jpg) | ![](showcase/lamp_clay.jpg) | ![](showcase/lamp_bedroom.jpg) | ![](showcase/lamp_cafe.jpg) |

- 模型：Khronos glTF Sample Assets · IridescenceLamp（https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/IridescenceLamp）
- 授权与署名：© 2022 Wayfair LLC，CC BY 4.0（https://creativecommons.org/licenses/by/4.0/）；模型与贴图 Eric Chadwick。
- 底图：Blender 4.3.2，Cycles 64 spp，960×540，AZ=-30 EL=0.55 FILL=0.8（3/4 左前方，坐姿视高略俯视；50 mm）。原样导入。
- 设置：环境／风格组每个场景 1 张主参考，产品造型与人物组不放；渲染模式 标准；严格构图锁定 / 结构控制图 / 局部结构修复 开；Identity Preserve：关闭（无 Logo）。
- 完整提示词、参考图、全部尝试和叠图：[`cases/lamp/prompt.md`](cases/lamp/prompt.md)

**产品外观（固定）**

```text
【产品外观】台灯：灰褐色（taupe）亚麻圆柱形灯罩，灯座为带幻彩（彩虹薄膜）光泽的透明球形玻璃，球内可见细金属管，亮银色金属灯颈与圆形底座。材质与颜色以图1的 Blender 材质为准。严格保持图1中台灯的轮廓、比例、画面位置和视角完全不变，灯罩形状与球形灯座不变，不添加文字或 Logo。
```

**极简卧室夜晚** · 环境参考 [`cases/lamp/refs/ref_bedroom.png`](cases/lamp/prompt.md)（见下图）（主参考；取：深夜室内、窗外深蓝夜色与微弱冷月光、台灯暖光晕、日式极简材质；不取：参考图里的物件位置。）

<img src="showcase/refs/lamp_bedroom.jpg" width="360" alt="参考图">

```text
【环境/风格】极简卧室夜晚：深夜，房间偏暗，台灯是主光源已点亮，灰褐色灯罩透出 2700K 暖黄光，在暖灰色微水泥墙和浅橡木床头柜面上形成柔和光晕；背景是米白亚麻床品的低矮床，薄纱窗帘外深蓝夜空，窗边微弱冷色月光；球形玻璃灯座折射暖光并带幻彩高光，玻璃球轮廓有清晰的亮边。安静、温馨的日式极简家居广告。
English: table lamp with taupe linen drum shade and iridescent glass globe base, switched on as the main light, on an oak nightstand in a dark minimalist Japandi bedroom late at night; keep the exact silhouette, scale, position and camera angle of image 1.
```

**咖啡馆** · 环境参考 [`cases/lamp/refs/ref_cafe.png`](cases/lamp/prompt.md)（见下图）（主参考；取：左侧傍晚窗光斜射、琥珀蜂蜜色调、浅景深、胡桃木桌面；不取：拿铁和绿植的具体位置（要求不遮挡台灯）。）

<img src="showcase/refs/lamp_cafe.jpg" width="360" alt="参考图">

```text
【环境/风格】咖啡馆：台灯放在胡桃木咖啡桌上，桌面一侧有拉花拿铁和小盆绿植（不遮挡台灯）；背景虚化的红砖墙、木架上的陶瓷杯、大窗和吊绿植，左侧傍晚阳光斜射进来，琥珀蜂蜜色调，浅景深；球形玻璃灯座折射窗光并带幻彩高光，底座在桌面有自然接触阴影。温暖的生活方式广告。
English: table lamp with taupe linen drum shade and iridescent glass globe base on a walnut café table, warm afternoon window light, latte and plant nearby; keep the exact silhouette, scale, position and camera angle of image 1.
```

| 场景 | 轮廓 IoU | 外轮廓 ≤3 px | 底部偏差 | 补充边缘 | 结果 |
|---|---|---|---|---|---|
| 极简卧室夜晚 | 66.8% | 47% | -199 px | 100% | ❌ |
| 咖啡馆 | 97.5% | 93% | -1 px | 94% | ✅ |

> 极简卧室夜晚：未达标原因：测量失败而不是错位。透明玻璃球灯座让 GrabCut 只保留了灯罩（底部 -199 px）；叠图上整盏灯与 Blender 轮廓逐像素重合，补充边缘指标 100%。第 1 次（白色灯罩提示词）按统一方法达标（IoU 96.4%），但灯罩颜色不忠实，所以没采用。

---

## 生成方式说明

案例三至六（以及单椅案例）的场景图由 Hark 的图像生成工具按插件的提示词结构生成：图 1 = Blender Camera Base，图 2 = 环境／风格主参考，提示词 = 产品外观 + 环境/风格 + 插件“严格构图锁定”会追加的构图硬规则。用来展示插件的工作流和预期效果，不是插件 Codex / Antigravity Provider 的直接输出。环境参考图同样由该工具生成，没有第三方版权素材。

## 致谢 / 模型授权

- 小米 YU7：Sketchfab，Ddiaz Design（展示用途，不分发）
- SheenChair：© 2020 Wayfair LLC，CC0 1.0
- MaterialsVariantsShoe：© 2021 Shopify，CC BY 4.0
- ChronographWatch：© 2025 Darmstadt Graphics Group GmbH，CC BY 4.0；Eric Chadwick（模型与贴图）；源自 “Chronograph Watch Mudmaster” by graphiccompressor（Sketchfab，CC BY 4.0）。表盘上的 Khronos / 3D Commerce / DGG 标志为商标
- AntiqueCamera：© 2018 UX3D，CC0 1.0；Maximilian Kamps。UX3D 标志为商标
- IridescenceLamp：© 2022 Wayfair LLC，CC BY 4.0；Eric Chadwick
- 以上 Khronos 模型均来自 https://github.com/KhronosGroup/glTF-Sample-Assets
