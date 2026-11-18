"""Fresh-process checks for backend probing and private C++ runtime state."""

import ctypes
import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def backend_marker(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """构建记录可选后端探测行为的原生标记模块。"""
    from setuptools import Distribution, Extension
    from setuptools.command.build_ext import build_ext

    root = Path(__file__).resolve().parents[1]
    output = tmp_path_factory.mktemp("backend_load_marker")
    extension = Extension(
        "_backend_load_marker",
        [str(root / "tests/native/backend_load_marker.cpp")],
        language="c++",
        py_limited_api=True,
        define_macros=[("Py_LIMITED_API", "0x030B0000")],
        extra_compile_args=["/std:c++17", "/utf-8"] if sys.platform == "win32" else ["-std=c++17"],
    )
    command = build_ext(Distribution({"ext_modules": [extension]}))
    command.build_lib, command.build_temp = str(output), str(output / "temp")
    command.ensure_finalized()
    command.run()
    return Path(command.get_ext_fullpath("_backend_load_marker"))


def subprocess_environment(package_parent: Path) -> dict[str, str]:
    """清理外部运行库变量，使子进程只使用指定安装包。"""
    environment = dict(os.environ)
    for name in ("LD_PRELOAD", "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "GGML_BACKEND_PATH", "MINERU_LLAMA_CPP_BACKEND"):
        environment.pop(name, None)
    environment["PYTHONPATH"] = str(package_parent)
    return environment


@pytest.mark.parametrize(
    "backend,layers,probes_gpu,error",
    [
        ("cpu", 99, False, "RuntimeError"),
        ("auto", 0, False, "RuntimeError"),
        ("vulkan", 0, False, "RuntimeError"),
        ("sycl", 0, False, "ValueError"),
        ("auto", 99, True, "RuntimeError"),
        ("vulkan", 99, True, "RuntimeError"),
    ],
)
def test_cpu_requests_skip_optional_backends(
    tmp_path: Path, backend_marker: Path, backend: str, layers: int, probes_gpu: bool, error: str
) -> None:
    """验证 CPU 请求和无效配置不会触发可选 GPU 后端探测。"""
    import mineru_llama_cpp

    source = Path(mineru_llama_cpp.__file__).resolve().parent
    package = tmp_path / "mineru_llama_cpp"
    shutil.copytree(source, package)
    prefix, suffix = ("", ".dll") if sys.platform == "win32" else ("lib", ".so")
    for name in ("vulkan", "metal"):
        shutil.copy2(backend_marker, package / "bin" / f"{prefix}ggml-{name}{suffix}")
    marker = tmp_path / "probed.txt"
    environment = subprocess_environment(tmp_path)
    environment.update(
        MINERU_LLAMA_CPP_BACKEND=backend,
        MINERU_TEST_BACKEND_MARKER=str(marker),
        GGML_BACKEND_PATH=str(package / "bin" / f"{prefix}ggml-vulkan{suffix}"),
    )
    script = """
from mineru_llama_cpp._mineru_llama_cpp import _EngineCore
import sys
try:
    _EngineCore('nonexistent-model.gguf', 'nonexistent-projector.gguf', 256, int(sys.argv[1]), 1, 0, 4)
except (RuntimeError, ValueError) as error:
    assert type(error).__name__ == sys.argv[2], repr(error)
    print('EXPECTED_ERROR', flush=True)
else:
    raise AssertionError('Missing model unexpectedly loaded')
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(layers), error], env=environment, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "EXPECTED_ERROR" in result.stdout
    assert marker.exists() is probes_gpu


@pytest.mark.skipif(sys.platform != "win32", reason="Windows native path encoding")
@pytest.mark.parametrize("folder", ["ascii", "中文", "🚀"], ids=["ascii", "chinese", "emoji"])
@pytest.mark.parametrize("layers", [0, 99], ids=["cpu-only", "all-backends"])
def test_windows_backend_install_path(tmp_path: Path, folder: str, layers: int) -> None:
    """从指定目录加载真实 wheel 的扩展和后端，两种发现模式都必须走到模型加载阶段。"""
    import mineru_llama_cpp

    install_dir = tmp_path / folder
    shutil.copytree(Path(mineru_llama_cpp.__file__).resolve().parent, install_dir / "mineru_llama_cpp")
    script = """
import sys
from pathlib import Path
from mineru_llama_cpp import Engine, _mineru_llama_cpp

assert Path(_mineru_llama_cpp.__file__).resolve().parent == Path.cwd() / "mineru_llama_cpp"
try:
    Engine("missing.gguf", "missing-mmproj.gguf", n_ctx_seq=256, n_gpu_layers=int(sys.argv[1]), n_parallel=1)
except RuntimeError as error:
    if str(error) != "load_model failed":
        raise
else:
    raise AssertionError("Missing model unexpectedly loaded")
"""
    subprocess.run(
        [sys.executable, "-c", script, str(layers)],
        cwd=install_dir,
        env=subprocess_environment(install_dir),
        check=True,
    )


def test_cpu_variant_selection_requires_scoring_entrypoint(tmp_path: Path) -> None:
    """验证可评分 CPU 变体能被选中，无评分模块不能冒充变体。"""
    import mineru_llama_cpp

    source = Path(mineru_llama_cpp.__file__).resolve().parent
    package = tmp_path / "mineru_llama_cpp"
    shutil.copytree(source, package)
    name = "ggml-cpu.dll" if sys.platform == "win32" else "libggml-cpu.so"
    cpu = package / "bin" / name
    if not cpu.exists():
        pytest.skip("Wheel already uses CPU variants instead of the base module")
    # 部分 ARM 构建仅提供基础模块；改名不会为其添加变体评分入口。
    library = ctypes.CDLL(str(source / "bin" / name))
    expected_error = (
        "load_model failed" if hasattr(library, "ggml_backend_score") else "Packaged CPU backend is unavailable"
    )
    cpu.rename(cpu.with_name(cpu.stem + "-test" + cpu.suffix))
    environment = subprocess_environment(tmp_path)
    environment["MINERU_LLAMA_CPP_BACKEND"] = "cpu"
    script = """
import sys
from mineru_llama_cpp._mineru_llama_cpp import _EngineCore
try:
    _EngineCore('nonexistent-model.gguf', 'nonexistent-projector.gguf', 256, 0, 1, 0, 4)
except RuntimeError as error:
    assert sys.argv[1] in str(error), str(error)
else:
    raise AssertionError('Missing model unexpectedly loaded')
"""
    result = subprocess.run(
        [sys.executable, "-c", script, expected_error], env=environment, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(sys.platform != "linux", reason="ELF runtime isolation is Linux-specific")
def test_private_runtime_symbols_are_not_exported(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证 Linux 库不会导出可能干扰进程的私有 C++ 运行库符号。"""
    import mineru_llama_cpp

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    from repair_linux_wheel import inspect_runtime_exports

    library = Path(mineru_llama_cpp.__file__).resolve().parent / "lib/libllama.so.1"
    assert inspect_runtime_exports(library) == set()


@pytest.mark.skipif(sys.platform != "linux", reason="LLVM regression reproduces Linux dynamic symbol binding")
@pytest.mark.parametrize("native_first", [True, False])
def test_native_and_llvm_import_order(native_first: bool) -> None:
    """验证原生包与系统 LLVM 的两种加载顺序均可正常完成。"""
    import mineru_llama_cpp

    directories = [Path("/usr/lib"), Path("/usr/lib64")]
    multiarch = sysconfig.get_config_var("MULTIARCH")
    if multiarch:
        directories.append(Path("/usr/lib") / multiarch)
    candidates = sorted({path.resolve() for directory in directories for path in directory.glob("libLLVM*.so*")})
    if not candidates:
        pytest.skip("System LLVM is not installed")
    package_parent = Path(mineru_llama_cpp.__file__).resolve().parent.parent
    script = """
import ctypes, sys
if sys.argv[2] == 'native-first':
    import mineru_llama_cpp
library = ctypes.CDLL(sys.argv[1])
import mineru_llama_cpp
print('IMPORTED', flush=True)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(candidates[-1]), "native-first" if native_first else "llvm-first"],
        env=subprocess_environment(package_parent),
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "IMPORTED" in result.stdout
