"""Deterministic stop/start ordering and fresh-process Engine destruction."""

import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import MMPROJ, MODEL


@pytest.fixture(scope="session")
def queue_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """链接已安装 wheel 的真实服务器队列，构建生命周期探针。"""
    if sys.platform not in {"linux", "darwin"}:
        pytest.skip("This probe requires exported ELF/Mach-O C++ server symbols")
    import mineru_llama_cpp

    root = Path(__file__).resolve().parents[1]
    upstream = root / "third_party/llama.cpp"
    libraries = Path(mineru_llama_cpp.__file__).resolve().parent / "lib"
    executable = tmp_path_factory.mktemp("queue_lifecycle") / "probe"
    command = [*shlex.split(os.environ.get("CXX", "c++")), "-std=c++17", "-pthread", "-UNDEBUG"]
    command += ["-I", str(root / "src/cpp")]
    for directory in ["tools/server", "common", "include", "ggml/include", "tools/mtmd", "vendor"]:
        command += ["-I", str(upstream / directory)]
    command += [
        str(root / "tests/native/queue_lifecycle_test.cpp"),
        "-L",
        str(libraries),
        "-l:libllama-server-impl.so",
        "-l:libllama-common.so.0",
        "-l:libllama.so.0",
        "-l:libmtmd.so.0",
        "-l:libggml.so.0",
        "-l:libggml-base.so.0",
        f"-Wl,-rpath,{libraries}",
        "-o",
        str(executable),
    ]
    if sys.platform == "darwin":
        command = [str(libraries / (arg[3:].replace(".so.0", ".0.dylib").replace(".so", ".dylib")))
                   if arg.startswith("-l:") else arg for arg in command]
    subprocess.run(command, check=True)
    return executable


@pytest.mark.parametrize("scenario", ["early-stop", "running-stop", "concurrent", "fatal-reader"])
def test_queue_termination_is_terminal(queue_probe: Path, scenario: str) -> None:
    # Test-only deadlock guard; this probe never loads a model or runs inference.
    """验证停止请求不会被延迟启动或并发启动覆盖。"""
    subprocess.run([str(queue_probe), scenario], check=True, timeout=30)


def test_fresh_process_immediate_close_without_generation() -> None:
    """在独立进程中验证模型初始化后立即关闭及重复关闭。"""
    if not Path(MODEL).is_file() or not Path(MMPROJ).is_file():
        pytest.skip("Real GGUF model and projector are required")
    import mineru_llama_cpp

    environment = dict(os.environ)
    environment.pop("LD_PRELOAD", None)
    environment.pop("LD_LIBRARY_PATH", None)
    environment["MINERU_LLAMA_CPP_BACKEND"] = "cpu"
    environment["PYTHONPATH"] = str(Path(mineru_llama_cpp.__file__).resolve().parent.parent)
    script = """
import sys
from mineru_llama_cpp import Engine
from mineru_llama_cpp._mineru_llama_cpp import _EngineCore
for _ in range(10):
    core = _EngineCore(sys.argv[1], sys.argv[2], 256, 0, 1, 0, 4)
    del core
for _ in range(5):
    engine = Engine(sys.argv[1], sys.argv[2], n_ctx_seq=256, n_gpu_layers=0, n_parallel=1, n_threads=4)
    engine.close()
    engine.close()
print('ALL_CLOSED', flush=True)
"""
    # Only bounds a constructor/destructor regression; no inference is performed.
    result = subprocess.run(
        [sys.executable, "-u", "-c", script, MODEL, MMPROJ],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ALL_CLOSED" in result.stdout
