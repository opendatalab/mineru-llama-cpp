"""不加载模型，验证生产后台循环的状态与诊断契约。"""

import os
import shlex
import subprocess
import sys
from pathlib import Path


def test_native_loop_failure_state(tmp_path: Path) -> None:
    """编译生产状态组件，覆盖真正线程抛错和意外正常退出。"""
    root = Path(__file__).resolve().parents[1]
    executable = tmp_path / (
        "lifecycle.exe" if sys.platform == "win32" else "lifecycle"
    )
    command = [
        *shlex.split(os.environ.get("CXX", "c++")),
        "-std=c++17",
        "-pthread",
        "-UNDEBUG",
    ]
    command += [
        "-I",
        str(root / "src/cpp"),
        str(root / "tests/native/engine_lifecycle_test.cpp"),
        "-o",
        str(executable),
    ]
    if sys.platform == "win32":
        command = [
            "cl",
            "/nologo",
            "/std:c++17",
            "/EHsc",
            "/utf-8",
            "/UNDEBUG",
            f"/I{root / 'src/cpp'}",
            str(root / "tests/native/engine_lifecycle_test.cpp"),
            f"/Fe:{executable}",
        ]
    subprocess.run(command, check=True, cwd=tmp_path)
    subprocess.run([str(executable)], check=True, timeout=10)
