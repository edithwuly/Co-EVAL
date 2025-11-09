import os
import subprocess
import tempfile
from typing import Tuple, Optional
import re
import js2py


def calculate_python_compiler(code):
    try:
        compile(code, '<string>', 'exec')
        return 1
    except:
        return 0


def calculate_c_compiler(code):
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = os.path.join(temp_dir, "test.c")
        with open(file_path, "w") as f:
            f.write(code)

        process = subprocess.run(
            ["gcc", file_path, "-o", os.path.join(temp_dir, "test")],
            capture_output=True,
            text=True
        )

        if process.returncode == 0:
            return 1
        else:
            return 0


def extract_class_name(source_code: str) -> Optional[str]:
    pattern = r'public\s+class\s+([A-Za-z0-9_]+)'
    match = re.search(pattern, source_code)

    if match:
        return match.group(1)
    return None


def calculate_java_compiler(source_code: str):
    class_name = extract_class_name(source_code)
    if not class_name:
        return 0

    # Create a temporary directory to store the Java file
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create the Java file
        file_path = os.path.join(temp_dir, f"{class_name}.java")
        with open(file_path, "w") as f:
            f.write(source_code)

        # Try to compile the Java file
        process = subprocess.run(
            ["javac", file_path],
            capture_output=True,
            text=True,
            cwd=temp_dir
        )

        # Check if compilation was successful
        if process.returncode == 0:
            return 1
        else:
            return 0


def calculate_cpp_compiler(source_code: str):
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = os.path.join(temp_dir, "test.cpp")
        with open(file_path, "w") as f:
            f.write(source_code)

        process = subprocess.run(
            ["g++", "-std=c++17", file_path, "-o", os.path.join(temp_dir, "test")],
            capture_output=True,
            text=True
        )

        if process.returncode == 0:
            return 1
        else:
            return 0


def calculate_golang_compiler(source_code: str):
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = os.path.join(temp_dir, "main.go")
        with open(file_path, "w") as f:
            f.write(source_code)

        process = subprocess.run(
            ["go", "build", file_path],
            capture_output=True,
            text=True,
            cwd=temp_dir
        )

        if process.returncode == 0:
            return 1
        else:
            return 0


def calculate_js_compiler(js_code):
    try:
        js2py.parse_js(js_code)
        return 1
    except Exception:
        return 0