# llama.cpp b12503 本地验证

2026-11-18，基于已合入 b12499 的项目主线 `7b09997`，在 Apple M6、macOS 27.2 上比较 b12499（`401563f10ec60754e688cd3165c1343cf4be9642`）与 b12503（`4577932fab595eaa8b27ad71734d5de1f3c07d8c`）。

两个上游标签相差四个提交，分别涉及 CUDA 卷积索引、V100 的 MoE 量化矩阵乘法、Web UI 的 Storybook，以及 OpenVINO 的模型支持与推理优化。本次 macOS wheel 使用 CPU 和 Metal。[上游提交对照](https://github.com/ggml-org/llama.cpp/compare/b12499...b12503)

## 构建与安装

- 在独立工作树干净构建 `mineru_llama_cpp-0.2.1-cp311-abi3-macosx_14_0_arm64.whl`。六个项目补丁均按序应用，第二次配置全部识别为已应用。wheel 的 SHA-256 为 `15b68fd735c0b3949352746fe8f6f7eb4438ce1f3b4fc2684afc17345eefda26`。
- 在源码目录外安装 wheel。CPython 3.11、3.12、3.13、3.14 均从已安装的原生扩展读取到 b12503 的完整提交号，并加载 CPU 与 Metal 后端。包内 11 个 Mach-O 通过 arm64 架构、最低系统版本和动态库依赖检查。
- 随包的 `llama-server --list-devices` 列出 Apple M6。CPU 模式完成 32-token 生成，视觉投影选择 CPU。三轮候选 Metal 测试均记录 25/25 层卸载并完成图文推理。
- 从已安装 wheel 运行完整 pytest，287 项通过，16 项按平台跳过，耗时 288.80 秒。跳过项包括 13 项 Linux CPU 拓扑用例与 3 项 ELF/LLVM 动态加载用例。Windows 和 Linux wheel 待跨平台验证。

## 真实页图对照

输入为 `demo1.pdf` 首页（1026 × 1400）、MinerU2.5-Pro-2605-1.2B Q8_0 主模型及配套 mmproj。已核对三份输入的 SHA-256。使用 `tools/benchmark_upgrade.py` 交替运行两版各三轮，每次使用独立进程，Python 3.14.4、单槽、贪心采样。文本生成与两阶段提取均先预热。

| 指标（三轮中位数） | b12499 | b12503 |
|---|---:|---:|
| 引擎初始化 | 0.414 秒 | 0.482 秒 |
| 32-token 文本生成 | 0.153 秒 | 0.153 秒 |
| 原尺寸图直送布局请求 | 9.565 秒 | 9.473 秒 |
| MinerU 两阶段提取 | 10.427 秒 | 10.388 秒 |
| 进程峰值 RSS | 1645.0 MiB | 1643.7 MiB |

六次运行的文本输出逐字一致。标准两阶段提取均得到相同的 19 个块，包含逐块内容、类别、坐标和阅读顺序。单页提取耗时范围分别为 10.247–10.455 秒和 10.266–10.600 秒。原尺寸图绕过 SDK 预处理直送布局请求时，两版的原始输出也逐字一致，均在 1024 tokens 上限截断。

本地原始数据位于 `build/validation-b12503/`。其中 `benchmark/round-*/` 保存各轮模型输出和原生日志，`benchmark/summary.json` 记录耗时及内存，`benchmark/output-comparison.json` 记录六次运行的逐项比较。
