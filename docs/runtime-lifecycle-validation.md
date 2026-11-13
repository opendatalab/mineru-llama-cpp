# 运行库加载与立即关闭修复验证

2026-10-22。基线 `b33f25dc`，继续固定 llama.cpp `86a283532072722c5f3363d37d59a874d09fa99b`。
本次为源码修复及隔离安装的本地候选 wheel 验证，未发布新版本或替换 MinerU 应用环境。

## 修复

1. Linux 的 libllama 保留静态 libstdc++，同时使用 `--exclude-libs,libstdc++.a` 隐藏其归档符号，
   避免 LLVM 等动态运行库的 `call_once` 函数与线程局部状态落入不同副本。
   wheel 审计新增私有运行库状态符号检查，仍保留原来的 manylinux ABI 下限。
2. Engine 在加载后端前校验请求。明确 CPU 或零 GPU 层数时，只加载 CPU 计算后端，
   复用上游 CPU 变体评分，并保留可选 BLAS。该路径不扫描 GPU 后端及通用 `GGML_BACKEND_PATH` 插件。
   自动/GPU 请求保留原来的发现行为；已注册的 GPU 不会被强制卸载。
3. `0005-preserve-early-server-stop.patch` 在同一把互斥锁下保留停止请求。
   延迟执行的 `start_loop()` 不再覆盖先发生的停止，正常销毁仍 join 后台线程。
   当前队列实例停止后不可隐式重启；Engine 每个实例已有的一次循环生命周期保持不变。
4. `0006-load-cpu-backend-without-gpu-probing.patch` 提供包内使用的 ggml CPU 加载入口。
   这是随配套原生库一起构建的接口，不改变 Python API；需要完整重建/安装 wheel。

CPU 按需加载改动位于 Engine；上游独立 llama-server 仍执行自己的设备发现，
但同样使用隔离运行库符号和修复后的队列。CPU 核心计数补丁继续保留。

## 验证结果

环境：Core Ultra 5 225H，Linux 7.0.0-38，Python 3.14.4，GCC 15.2，CPU + Vulkan 构建。

| 检查 | 结果 |
| --- | --- |
| 完整 CPU 测试集，无 LD_PRELOAD | 185 passed，46 skipped |
| 明确 CPU、零 GPU 层数、无效后端 | 临时探针确认 GPU 后端未被调用；无效配置仍报错 |
| 自动/显式 GPU 请求 | 可选后端仍被探测；不可用时按原规则处理 |
| 无基础 CPU MODULE、仅有 CPU 变体 | 成功选中 CPU 变体 |
| 先导入原生包/先加载系统 LLVM | 两种顺序均正常，无预加载 |
| 停止先于启动、运行中停止、并发启动/停止 | 三种原生队列检查通过，包含重复停止和终止后的延迟入口 |
| 即时关闭 | 独立进程内 10 次原生 core 销毁、5 次公开 Engine.close 通过，无预先生成或 sleep |
| 常规行为 | 真实图像识别、流式、异步、错误请求恢复、重复初始化拒绝、关闭后重建及内存回归通过 |
| ELF 私有运行库符号审计 | wheel 中全部 11 个 ELF 均通过 |
| 补丁重复配置 | 六个补丁均识别为 already applied |
| Ruff、git diff --check | 通过 |

死锁回归中的时间保护仅约束不执行推理的队列及构造/销毁测试，防止坏版本挂住 CI；
真实推理验证没有添加推理墙钟超时。46 个跳过项为当前平台不适用的检查。

## Vulkan 验证

使用当前真实权重及 demo1 第一页的 1036 x 1036 布局图，生成 32 token 布局前缀。
新 llama-server 在没有 LD_PRELOAD、没有限制 Vulkan ICD 的条件下加载 Intel ARL (`8086:7d51`)，
主模型 25/25 层及视觉编码器均使用 GPU。输出与 CPU 视觉参考前缀一致，进程正常退出，内核无 GPU 错误。

另通过公开 Python Engine 的 GPU 初始化、短文本生成和关闭检查，同样没有 LD_PRELOAD 或 ICD 过滤。
两项 GPU 检查均保留本机 Xe-LPG 需要的 `GGML_VK_SERIALIZE_SUBMISSIONS=1`；
本次运行库和生命周期修复不替代该 GPU 图提交规避策略。

## 产物与范围

本地候选为 `build/runtime-lifecycle-final/wheels/mineru_llama_cpp-0.1.3-cp310-abi3-linux_x86_64.whl`，
版本号仍为未发布源码候选的 0.1.3。它不是经过 manylinux 发布矩阵认证的发行包。
跨平台真机及 manylinux x86_64/ARM64 发布构建仍需 CI，不能由本机结果代替。
推理进行期间调用 close 的并发契约未扩展。

结构化结果、wheel 哈希及 ELF 数据在 `build/runtime-lifecycle-final/validation.json`；
完整测试日志为 `tests.log`，GPU 日志位于 `gpu-smoke/` 和 `gpu-engine.log`，重复配置日志为 `reconfigure.log`。
此前的根因堆栈、符号绑定和隔离实验保留在 `build/runtime-lifecycle-diagnosis/`。


## 0.2.1 故障状态契约补充

历史验证只覆盖正常 shutdown；本轮新增循环退出锁存、明确关闭异常和异步取消排空。
`tests/test_engine_lifecycle_state.py` 在真实 C++ 线程抛出异常，验证首次原因和终止状态；
`tests/test_engine_failure_contract.py` 验证公开 Python 错误及取消规则；实际模型验收脚本位于
MinerU 的 `scripts/acceptance/llama_cpp_lifecycle.py`。合成异常、真实引擎显式关闭与整个进程退出
必须分开记录，不能据此宣称已经修复所有原生崩溃根因。
