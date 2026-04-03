import subprocess
import sys

import os

SCRIPTS = [
    "01_capture_image.py",
    "02_maze_warp_from_json.py",
    "03_maze_circles_and_grid.py",
    "04_solve_maze.py",
    "05_unwrap_and_overlay_path.py",
    "convert_points.py"
]

UNWARPED_JSON = "part_2_maze_solution/solution_path_points_unwarped.json"

def run_step(script_name):

    print(f"Running: {script_name}\n")

    result = subprocess.run(
        [sys.executable, script_name],
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    if result.returncode != 0:
        print(f"ERROR: {script_name} failed with return code {result.returncode}")
        sys.exit(1)

def main():
    for script in SCRIPTS:
        run_step(script)

if __name__ == "__main__":
    main()