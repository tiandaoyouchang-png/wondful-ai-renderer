# PC feedback hardening 验收记录（2026-10-23）

修复分支：`fix/pc-feedback-hardening`。基线：`4c81a6a925740cea7231d36c8241f4590cdd234f`。
结论：本轮是经过离线与 Blender CPU 验收的测试候选包，不是稳定版发布。
未修改版本号、自动更新清单、main 或正式发布资产。

## 已完成的代码修正

- 保留“没有高度证据，不创建凸起路沿”原则。纠正旧测试对纯色平面条带的错误预期；增加实测 12 cm 高度的正向回归。
- 地面填充只接受朝上的地面；垂直路沿面只在已测量的凸起条带内接受。低墙和朝下平面不再仅因高度接近地面而被填色。
- 多视角投影缓存校验主图、主相机、几何类型、主视角通道、场景分析和相关输出设置。
  场景快照包括对象变换、相机镜头、网格坐标及拓扑、修改器求值后的网格。旧格式缓存及缺少任一通道的缓存不复用。
- 世界光照缓存增加主相机与模型身份，避免同一张主图换相机后复用旧世界方向。
- 直连图像重试、代理回退、CLI 模型回退共享单调时钟预算；代理只接收剩余时间，不重新开始完整超时。
- 流式心跳不能无限延长图像请求；超时停止重试及代理回退。取消前置检查覆盖直连和代理入口。
- CLI 提示词通过标准输入发送，执行过程纳入已存在的可取消进程管理。小于 1 秒的剩余超时不再被截断为 0（无限等待）。
- 街景历史样本缺失时明确跳过；保留合成街景回归，避免依赖另一台机器的绝对路径。

- 全量复测曾发现一次 Blender 原生崩溃：分层 NumPy 工作线程与下一视角渲染重叠。自动操作链现在等待分层任务结束，再进入下一阶段；端到端测试明确验证下一视角启动时 pending=0。原始失败与崩溃记录保留在 evidence 中。该修改控制已观察到的并发触发条件，不宣称已排除所有原生崩溃原因。

## 验证方式与范围

最终全量结果见同目录 `pc-hardening-results.json`；原始日志在交付包的 evidence 目录。

1. Python 编译、undefined-name 检查、git diff 空白检查。
2. 全量 unittest：Blender 4.3.2 加入 PATH，设置 WONDFUL_BLENDER 与本轮 WONDFUL_ZIP。
3. `tools/blender_acceptance.py`：9 项全部通过，使用 Cycles CPU 640×360；包括原生深度/法线/Object Index、可见徽标和车牌、遮罩、身份回贴和状态恢复。
4. `tools/pc_hardening_acceptance.py`：10 项全部通过，包括网格/相机/修改器缓存失效、仅凸起条带建几何、独立路沿材质、辅助网格/材质清理、纯场景白模与几何通道、场景状态恢复。
5. `tools/e2e_render_views_bg.py -- ZIP OUT pure-scene`：纯场景双视角操作链成功；产品遮罩全空、跳过产品验收、两张图可切换，输出目录仅有图片。
6. 全量 unittest 中包含带产品的双视角、分层 PSD 和干净输出目录流程。
7. 新增 15 项专项回归，覆盖缓存、路沿、墙面、取消、标准输入、模型回退、剩余超时和流式心跳。

Blender 白模、几何通道及材质操作是真实运行；AI 返回使用本地模拟后端，多视角 AI 编辑返回投影底图。
这些检查证明插件流程与数据约束，不证明真实模型的视觉质量。本轮未调用用户真实生图账户。

## 未验证及仍需收口

- 历史实拍/生成样本和部分 MoGe 测量文件未随仓库提供，相关测试标为跳过。
- 未验证 Windows、macOS、Blender 5.1、EEVEE/GPU 和交互式窗口操作；当前实测环境为 Linux / Blender 4.3.2 / Cycles CPU。
- 未实测真实后端额度、计费、限流、长耗时生图以及用户原始 216° 问题场景的最终视觉效果。
- “最多生成次数”仍限制外层生成/验收轮数；底层网络重试、代理回退及局部修复尚未统一成一个全局物理请求计数器，不能把该设置当作严格的后端请求次数或计费上限。
- 单调时钟预算已经约束图像重试/回退；登录刷新、模型目录查询、环境文本分析等准备阶段仍有各自的超时，不能宣称整个点击任务绝对不超过该秒数。
- 已建立连接的流式读取支持取消和超时中断；DNS/TLS/连接建立阶段仍依赖底层网络超时。
- 缓存快照未覆盖所有第三方节点、实例化机制、外部资源和材质节点状态，复杂场景仍需实测；旧缓存会保守失效。

因此，这份包适合本地复核和继续收口上述问题，不应标记为全部修复完成或稳定版。

## 重现

```bash
python -m compileall -q wondful_ai_renderer tests tools
python tools/check_undefined_names.py
python tools/build_release.py --no-manifest
# 配置 PATH、WONDFUL_BLENDER 和 WONDFUL_ZIP 后：
python -m unittest discover -s tests -p 'test_*.py' -v
WONDFUL_STRUCTURE_ENGINE=CYCLES blender -b --factory-startup --python-exit-code 1 --python tools/blender_acceptance.py -- --out /tmp/wondful_accept
blender -b --factory-startup --python-exit-code 1 --python tools/pc_hardening_acceptance.py -- /tmp/wondful_pc_accept
blender -b --factory-startup --python-exit-code 1 --python tools/e2e_render_views_bg.py -- /absolute/addon.zip /tmp/wondful_pure pure-scene
```
