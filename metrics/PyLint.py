import os
import subprocess
import tempfile


def calculate_pylint(code):
    score = 0
    try:
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp_file:
            temp_file.write(code.encode('utf-8'))
            temp_file_path = temp_file.name

        result = subprocess.run(['pylint', temp_file_path], capture_output=True, text=True)

        print("Pylint Output:\n", result.stdout)

        score_line = [line for line in result.stdout.splitlines() if "Your code has been rated at" in line]
        if score_line:
            score = score_line[0].split()[6].split('/')[0]
    except Exception as e:
        print(f"Error running Pylint: {e}")
    finally:
        os.remove(temp_file_path)
    return score