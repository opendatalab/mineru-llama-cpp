"""验证真实打包加载器隔离可选后端的驱动初始化异常。"""
import ctypes
from pathlib import Path
import sys


def test_optional_backend_failure_keeps_cpu_available(tmp_path):
    """加载抛异常的真实 MODULE 后，注册表保持原状且 CPU 仍可使用。"""
    from setuptools import Distribution, Extension
    from setuptools.command.build_ext import build_ext
    import mineru_llama_cpp

    root = Path(__file__).resolve().parents[1]
    extension = Extension("_optional_backend_test", [str(root / "tests/native/optional_backend_binding.cpp")],
                          language="c++", py_limited_api=True, define_macros=[("Py_LIMITED_API", "0x030B0000")],
                          extra_compile_args=["/std:c++17", "/utf-8"] if sys.platform == "win32" else ["-std=c++17"])
    command = build_ext(Distribution({"ext_modules": [extension]}))
    command.build_lib, command.build_temp = str(tmp_path), str(tmp_path / "temp")
    command.ensure_finalized()
    command.run()
    module = command.get_ext_fullpath("_optional_backend_test")
    package = Path(mineru_llama_cpp.__file__).resolve().parent
    name = "ggml.dll" if sys.platform == "win32" else "libggml.1.dylib" if sys.platform == "darwin" else "libggml.so.1"
    library_path = next(package / directory / name for directory in ("lib", "bin") if (package / directory / name).exists())
    library = ctypes.CDLL(str(library_path))
    library.ggml_backend_load_all_from_path.argtypes = [ctypes.c_char_p]
    library.ggml_backend_load_all_from_path.restype = None
    library.ggml_backend_load.argtypes = [ctypes.c_char_p]
    library.ggml_backend_load.restype = ctypes.c_void_p
    library.ggml_backend_reg_count.restype = ctypes.c_size_t
    library.ggml_backend_dev_by_type.argtypes = [ctypes.c_int]
    library.ggml_backend_dev_by_type.restype = ctypes.c_void_p
    library.ggml_backend_load_all_from_path(str(package / "bin").encode())
    before = library.ggml_backend_reg_count()
    assert library.ggml_backend_load(str(module).encode()) is None
    assert library.ggml_backend_reg_count() == before
    assert library.ggml_backend_dev_by_type(0), "CPU must remain registered after optional initialization fails"
