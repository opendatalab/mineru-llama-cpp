"""在仓库外验证已安装扩展及其打包的 CPU/GPU 后端，无需模型。"""

import ctypes
import json
from pathlib import Path
import sys
import os
import subprocess


def main() -> None:
    """加载真实安装包的后端并要求至少有一个可用 CPU 设备。"""
    if "--without-external-gpu-runtime" in sys.argv:
        # 在新进程中移除构建工具链搜索路径，验证 CPU 不依赖外部 GPU 运行库。
        environment = os.environ.copy()
        for name in ("VULKAN_SDK", "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH"):
            environment.pop(name, None)
        environment["PATH"] = os.pathsep.join(path for path in environment.get("PATH", "").split(os.pathsep)
                                              if "vulkansdk" not in path.lower())
        subprocess.run([sys.executable, str(Path(__file__).resolve())], env=environment, check=True)
        return
    import mineru_llama_cpp
    from mineru_llama_cpp import _mineru_llama_cpp as native

    package = Path(mineru_llama_cpp.__file__).resolve().parent
    name = "ggml.dll" if sys.platform == "win32" else "libggml.1.dylib" if sys.platform == "darwin" else "libggml.so.1"
    # Windows 的共享 DLL 按 RUNTIME 安装到 bin，Unix 的共享库安装到 lib。
    library_path = next((package / directory / name for directory in ("lib", "bin")
                         if (package / directory / name).is_file()), None)
    assert library_path is not None, f"packaged {name} missing under {package}"
    library = ctypes.CDLL(str(library_path))
    library.ggml_backend_load_all_from_path.argtypes = [ctypes.c_char_p]
    library.ggml_backend_load_all_from_path.restype = None
    library.ggml_backend_dev_by_type.argtypes = [ctypes.c_int]
    library.ggml_backend_dev_by_type.restype = ctypes.c_void_p
    library.ggml_backend_load_all_from_path(str(package / "bin").encode())
    # ggml_backend_dev_type 中 CPU、GPU 的值分别为 0、1。
    cpu = bool(library.ggml_backend_dev_by_type(0))
    gpu = bool(library.ggml_backend_dev_by_type(1) or library.ggml_backend_dev_by_type(2))
    assert cpu, "packaged CPU backend is not available"
    print(json.dumps({"package": str(package), "extension": str(Path(native.__file__).name),
                      "llama_cpp_commit": native._llama_cpp_commit,
                      "cpu_available": cpu, "gpu_available": gpu,
                      "gpu_model_test": "not run (model-free installation check)"}))
    if not gpu:
        print("SKIP GPU execution: no GPU exposed by this runner; CPU/import checks passed")


if __name__ == "__main__":
    main()
