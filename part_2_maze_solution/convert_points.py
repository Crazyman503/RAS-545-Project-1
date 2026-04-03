import json
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(script_dir, "part_2_maze_solution", "solution_path_points_unwarped.json")

with open(json_path) as f:
    data = json.load(f)

pixel_coords = [(p[0], p[1]) for p in data["unwarped_path_pixels"]]

print("pixel_coords = [")
for i, coord in enumerate(pixel_coords):
    comma = "," if i < len(pixel_coords) - 1 else ""
    print(f"{coord}{comma}")
print("]")