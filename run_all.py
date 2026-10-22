"""Run every section in order with the ArcGIS Pro Python environment.

    "C:\Program Files\ArcGIS\Pro\bin\Python\envs\arcgispro-py3\python.exe" run_all.py [s1 s2 s3 s4]
"""
import subprocess
import sys
from pathlib import Path

SCRIPTS = {
    "s1": ["src/s1_data_engineering.py", "src/s1_visualization.py"],
    "s2": ["src/s2_prediction.py"],
    "s3": ["src/s3_suitability.py"],
    "s4": ["src/s4_patterns_spacetime.py"],
}
root = Path(__file__).resolve().parent
wanted = sys.argv[1:] or list(SCRIPTS)
for key in wanted:
    for script in SCRIPTS[key]:
        print(f"=== {script}", flush=True)
        subprocess.run([sys.executable, str(root / script)], check=True, cwd=root)
