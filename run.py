import subprocess
import sys
import os 

venv_python = os.path.join("venv", "bin", "python")
streamlit_cli = os.path.join("venv", "bin", "streamlit")

# List of Python scripts to run with their arguments

cmds = [
    [venv_python, "profiler.py", "data/data.csv"],
    [venv_python, "llm_req.py"],
    [streamlit_cli, "run", "dashboard_st.py"], 
]

for cmd in cmds:
    print(f"\nRunning {cmd} ...")
    try:
        # Prepend the Python executable
        result = subprocess.run(cmd, capture_output=True, text=True)

        # Print stdout and stderr
        if result.stdout:
            print("Output:")
            print(result.stdout)
        if result.stderr:
            print("Errors:")
            print(result.stderr)

        # Check return code
        if result.returncode == 0:
            print(f"{cmd} executed successfully ✅")
        else:
            print(f"{cmd} failed with return code {result.returncode} ❌")

    except Exception as e:
        print(f"Failed to run {cmd}: {e}")
