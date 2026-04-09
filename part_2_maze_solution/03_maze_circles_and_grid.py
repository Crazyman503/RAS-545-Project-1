# #!/usr/bin/env python3
# """
# maze_circles_and_grid.py

# Workflow:
#   1) Detect circles via HoughCircles and classify color (red/green/unknown).
#      Saves a circles overlay.
#   2) Build grid over the maze, detect black walls, and compute per-cell occupancy:
#         value = 1 -> no wall in the cell
#         value = 0 -> wall present in the cell (>= threshold % of wall pixels)
#      Saves:
#         - grid overlay (lines only)
#         - annotated grid overlay (0/1 at cell centers)
#         - walls mask (255 = wall)
#   3) Write a single JSON with:
#         - circles: [ {center:[x,y], radius:r, color:"red/green/unknown"} ]
#         - grid_size_px, grid_rows, grid_cols, threshold_percent
#         - paths to overlays and mask
#         - cells: [ {row, col, value, center_px:[x,y]} ]

# Usage:
#   python maze_circles_and_grid.py maze.png \
#     --grid 30 --threshold 0.0 \
#     --circles-overlay-out circles_overlay.png \
#     --grid-overlay-out grid_overlay.png \
#     --grid-overlay-annot-out grid_overlay_annot.png \
#     --walls-mask-out walls_mask.png \
#     --json-out result.json \
#     --adaptive 0 --blur 5 --open 0 --close 0 \
#     --font-scale 0.4 --thickness 1
# """

#!/usr/bin/env python3
import argparse
import sys
import json
import numpy as np
import cv2

# ---------------- Circle color detection ----------------

def detect_color(frame, center, radius):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)

    # Very wide red ranges (dark to bright)
    red_mask1 = cv2.inRange(hsv, np.array([0, 50, 20]), np.array([15, 255, 255]))
    red_mask2 = cv2.inRange(hsv, np.array([160, 50, 20]), np.array([180, 255, 255]))
    red_mask = red_mask1 | red_mask2

    # Very wide green range (dark to bright)
    green_mask = cv2.inRange(hsv, np.array([25, 30, 20]), np.array([95, 255, 255]))

    red_count = cv2.countNonZero(cv2.bitwise_and(red_mask, mask))
    green_count = cv2.countNonZero(cv2.bitwise_and(green_mask, mask))
    total = cv2.countNonZero(mask)
    threshold = 0.1 * total

    if green_count > threshold and green_count > red_count:
        return "green"
    elif red_count > threshold and red_count > green_count:
        return "red"
    return None

def detect_circles_and_overlay(img_bgr, overlay_path):
    overlay = img_bgr.copy()
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (9, 9), 2)

    circles = cv2.HoughCircles(
        gray_blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
        param1=80, param2=35, minRadius=10, maxRadius=60
    )

    circles_info = []

    if circles is not None:
        circles = np.round(circles[0]).astype(int)
        for (x, y, r) in circles:
            color_label = detect_color(img_bgr, (x, y), r)
            label = color_label if color_label else "unknown"
            circles_info.append({
                "center": [int(x), int(y)],
                "radius": int(r),
                "color": label
            })
            cv2.circle(overlay, (x, y), r, (0, 255, 0), 3)
            cv2.circle(overlay, (x, y), 3, (0, 255, 0), -1)
            cv2.putText(overlay, label, (x - r, y - r - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imwrite(overlay_path, overlay)
    return circles_info

# ---------------- Grid + walls ----------------

def binarize_walls(gray: np.ndarray, adaptive: bool) -> np.ndarray:
    if adaptive:
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 25, 8
        )
    else:
        threshold_value, binary = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY_INV)
    return binary

def morph(mask: np.ndarray, k_open: int, k_close: int) -> np.ndarray:
    m = mask.copy()
    if k_open > 0:
        ko = cv2.getStructuringElement(cv2.MORPH_RECT, (2*k_open+1, 2*k_open+1))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, ko, iterations=1)
    if k_close > 0:
        kc = cv2.getStructuringElement(cv2.MORPH_RECT, (2*k_close+1, 2*k_close+1))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kc, iterations=1)
    return m

def draw_grid_lines(img: np.ndarray, grid: int) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    color = (255, 255, 255)
    for y in range(0, h, grid):
        cv2.line(out, (0, y), (w-1, y), color, 1)
    for x in range(0, w, grid):
        cv2.line(out, (x, 0), (x, h-1), color, 1)
    return out

def draw_grid_with_values(img: np.ndarray, grid: int, values_mat: np.ndarray, font_scale: float, thickness: int) -> np.ndarray:
    out = draw_grid_lines(img, grid)
    h, w = out.shape[:2]
    gh, gw = values_mat.shape
    for gy in range(gh):
        y0 = gy * grid
        y1 = min((gy + 1) * grid, h)
        cy = int((y0 + y1) / 2)
        for gx in range(gw):
            x0 = gx * grid
            x1 = min((gx + 1) * grid, w)
            cx = int((x0 + x1) / 2)
            text = str(int(values_mat[gy, gx]))
            cv2.putText(out, text, (cx-6, cy+5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), thickness+2, cv2.LINE_AA)
            cv2.putText(out, text, (cx-6, cy+5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), thickness, cv2.LINE_AA)
    return out

# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default = "part_2_maze_solution/maze_warp.png")
    ap.add_argument("--grid", type=int, default=23)
    ap.add_argument("--adaptive", type=int, default=0)
    ap.add_argument("--blur", type=int, default=5)
    ap.add_argument("--open", type=int, default=0)
    ap.add_argument("--close", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=1)

    ap.add_argument("--circles-overlay-out", default="part_2_maze_solution/circles_overlay.png")
    ap.add_argument("--grid-overlay-out", default="part_2_maze_solution/grid_overlay.png")
    ap.add_argument("--grid-overlay-annot-out", default="part_2_maze_solution/grid_overlay_annot.png")
    ap.add_argument("--walls-mask-out", default="part_2_maze_solution/walls_mask.png")
    ap.add_argument("--json-out", default="part_2_maze_solution/result.json")

    ap.add_argument("--font-scale", type=float, default=0.4)
    ap.add_argument("--thickness", type=int, default=1)

    args = ap.parse_args()

    img = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Error: cannot read image '{args.input}'", file=sys.stderr)
        sys.exit(1)

    grid = max(1, args.grid)

    # 1. Detect circles
    circles_info = detect_circles_and_overlay(img, args.circles_overlay_out)

    # 2a. Grid overlay
    grid_overlay = draw_grid_lines(img, grid)
    cv2.imwrite(args.grid_overlay_out, grid_overlay)

    # 2b. Walls mask
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if args.blur > 0:
        k = max(1, args.blur | 1)
        gray = cv2.medianBlur(gray, k)
    walls_mask = binarize_walls(gray, adaptive=bool(args.adaptive))
    walls_mask = morph(walls_mask, k_open=max(0, args.open), k_close=max(0, args.close))

    # RASE walls where circles are
    for c in circles_info:
        cx, cy = int(c["center"][0]), int(c["center"][1])
        r = int(c["radius"] * 1.2)
        cv2.circle(walls_mask, (cx, cy), r, 0, -1)  # set to 0 (non-wall)

    cv2.imwrite(args.walls_mask_out, walls_mask)

    # 2c. Build grid cells
    h, w = walls_mask.shape[:2]
    gh = (h + grid - 1) // grid
    gw = (w + grid - 1) // grid
    values_mat = np.zeros((gh, gw), dtype=np.uint8)
    cells = []
    for gy in range(gh):
        y0 = gy * grid
        y1 = min((gy + 1) * grid, h)
        cy = int((y0 + y1) / 2)
        for gx in range(gw):
            x0 = gx * grid
            x1 = min((gx + 1) * grid, w)
            cx = int((x0 + x1) / 2)
            blk = walls_mask[y0:y1, x0:x1]
            wall_pct = (blk > 0).mean() * 100.0 if blk.size > 0 else 0.0
            value = 0 if wall_pct >= args.threshold else 1
            values_mat[gy, gx] = value
            cells.append({
                "row": int(gy),
                "col": int(gx),
                "value": int(value),
                "center_px": [int(cx), int(cy)]
            })

    # 2d. Annotated grid
    grid_overlay_annot = draw_grid_with_values(img, grid, values_mat, args.font_scale, args.thickness)
    cv2.imwrite(args.grid_overlay_annot_out, grid_overlay_annot)

    # 3. Write JSON
    meta = {
        "input": args.input,
        "circles_overlay_path": args.circles_overlay_out,
        "grid_size_px": grid,
        "grid_rows": int(gh),
        "grid_cols": int(gw),
        "threshold_percent": args.threshold,
        "grid_overlay_path": args.grid_overlay_out,
        "grid_overlay_annot_path": args.grid_overlay_annot_out,
        "walls_mask_path": args.walls_mask_out,
        "circles": circles_info,
        "cells": cells
    }
    with open(args.json_out, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Circles overlay saved to: {args.circles_overlay_out}")
    print(f"Grid overlay saved to: {args.grid_overlay_out}")
    print(f"Walls mask saved to: {args.walls_mask_out}")
    print(f"Annotated grid overlay saved to: {args.grid_overlay_annot_out}")
    print(f"JSON saved to: {args.json_out}")
    print("Legend: cell value 1 = path, 0 = wall")

if __name__ == "__main__":
    main()
