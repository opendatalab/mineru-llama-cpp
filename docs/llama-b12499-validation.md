# llama.cpp b12499 本地升级验证

2026-11-18。在项目 v0.2.1（`bb9bbc4233d2c20c2944b16d2859454a9c5194b0`）上，分别用同一台 Apple M6、macOS 27.2（26B123）、AppleClang 21、SDK 27.2 和 Python 3.14.4 从源码构建基线和候选 wheel。两个包都以 macOS 14 为部署目标，包版本均为 0.2.1。可从已安装扩展的 `_llama_cpp_commit` 区分二者。

| 构建 | llama.cpp 提交 |
|---|---|
| 基线 b11524 | `86a283532072722c5f3363d37d59a874d09fa99b` |
| 候选 b12499 | `401563f10ec60754e688cd3165c1343cf4be9642` |

## 构建与引擎

- 干净子模块首次配置应用六个补丁，重复配置识别全部六个。混合 CPU 拓扑补丁保留了上游新增的 Linux RISC-V 分支。CLI 图文任务在服务器线程分词后计算消息边界，等待空闲槽位时复用已经生成的 tokens。
- 在源码树外安装候选 wheel，并在 CPython 3.11、3.12、3.13、3.14 上导入同一个 ABI3 扩展，逐个检查提交、CPU 和 Metal 后端。wheel 含一个 Python 扩展，11 个 Mach-O 均为 arm64，通过最低系统版本、RPATH 和动态库依赖审计。
- 打包的 `llama-server --list-devices` 能列出 Apple M6。独立进程以 CPU 完成文本生成，日志确认视觉投影也选择 CPU。Metal 三轮均记录模型 25/25 层卸载，并完成图文推理。
- 最终安装包的整套 pytest 为 **287 passed，16 skipped**（262.49 秒）。使用真实 GGUF、mmproj 及页图，覆盖并发超过槽位数、非法输入与 UTF-8、流式边界、取消、关闭和会话采样约束。Linux 混合 CPU 拓扑 13 项及 ELF/LLVM 3 项按平台跳过。Windows 和 Linux wheel 待跨平台验证。

## 真实页图的三轮对照

输入为 Magic-PDF 的 `demo1.pdf` 首页（1026×1400），主模型和投影均为本机 MinerU2.5-Pro-2605-1.2B Q8_0。使用 `mineru-vl-utils 2.0.8`，同一 Python 解释器从独立安装目录加载两版 wheel。`tools/benchmark_upgrade.py` 按基线/候选、候选/基线、基线/候选的顺序运行，每次启动独立进程，文本生成与两阶段提取先预热。

| 指标（三轮中位数） | b11524 | b12499 | 变化 |
|---|---:|---:|---:|
| 引擎初始化 | 0.711 s | 0.380 s | −46.6% |
| 32-token 文本生成 | 0.165 s | 0.166 s | +1.0% |
| 原尺寸图直送布局提示词 | 9.499 s | 9.976 s | +5.0% |
| MinerU 两阶段提取 | 11.021 s | 10.876 s | −1.3% |
| 进程峰值 RSS | 1643.8 MiB | 1644.8 MiB | +0.06% |

**标准 MinerU 流程。** 三轮文本生成逐字一致。两阶段提取每次均为 19 块，每块类别、坐标、顺序、角度和跨块标记一致。18 块内容逐字一致。零基编号 7 的作者行仅改变 LaTeX 上标及逗号周围的空格。逐项对照原 PDF 和标框页图，四位作者姓名、单位上标、通信作者星号和标点均一致。另用 `MinerUClient.layout_detect` 捕获实际经 1036×1036 预处理的布局输出。两版各跑三次，六次原始布局文本逐字一致，解析后的 19 个框也完全相同。

**原尺寸图片直送 Engine。** 诊断脚本另有一条路径，直接对 1026×1400 页图请求布局。三轮基线均输出表格标记和页面文字，生成 884 tokens 并停止。候选三轮均输出 46 个完整的全页宽 header 框，在 1024 tokens 上限截断。原页有期刊页眉、标题、摘要和脚注，两种输出都无法还原版面。这一行耗时包含不同的生成长度。上表仅反映本机这一页和这套模型的结果。

## 本地产物

验证文件位于升级工作树的 `build/validation-b12499/`。

- `candidate/wheels/` 含最终候选包。`candidate/final-package-check.json` 记录 wheel SHA-256、实际导入的扩展路径、上游提交，以及性能对照使用的安装包与最终包内 19 个运行文件逐字节一致的检查结果。
- `candidate/tests-final.log`、`candidate/wheel-audit.json`、`candidate/smoke-3.*.json`、`candidate/server-devices.log` 和 `candidate/cpu/` 保存各项验收结果。
- `benchmark/summary.json` 和 `benchmark/round-*/{baseline,candidate}/result.json` 保存三轮测量、原始文本和两阶段块。`inputs/SHA256SUMS` 固定本次页图及两个 GGUF 的内容。
- `review/round-comparison.json` 逐项记录输出差异。`review/sdk-{baseline,candidate}.json` 保存标准布局入口的原始输出。`review/blocks.png` 与 `review/author-line.png` 用于核对版面及作者行。
