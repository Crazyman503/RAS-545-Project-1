import numpy as np
import cv2
import time
import pydobot
from camera_utilities import apply_affine, fit_affine, apply_homography, fit_homography
from robot_utilities_2 import move_to_home, move_to_specific_position, get_current_pose

M = np.array([
    [3.81562905e-04, -4.82264725e-01,  4.13297105e+02],
    [-4.58721299e-01,  3.58521982e-04,  1.36198619e+02]
], dtype=np.float64)

H = np.array([
    [-2.85534558e-02, -4.09196707e-01,  4.12237951e+02],
 [-5.05804158e-01,  2.08688321e-03,  1.49980827e+02],
 [-9.76721146e-05,  2.26047821e-04,  1.00000000e+00]]
    , dtype=np.float64)
M[0, 2] -= 13.0  # shift +/-X
M[1, 2] -= 2.0  # shift +/- y
def move_robot_point(device,M,u,v):
    Xa, Ya = apply_affine(M, u, v) # Using Affine
    # Xa, Ya = apply_homography(H, u, v) # Using Homography
    print(f"Affine:  pixel({u:.3f}, {v:.3f}) -> robot({Xa:.6f}, {Ya:.6f})")
    move_to_specific_position(device, x=Xa, y=Ya, z=-45)
    time.sleep(1)

def main():
    device = pydobot.Dobot(port="COM7")
    device.speed(100, 100)
    move_to_home(device)
    time.sleep(1)

    # Example pixel coordinates from clicks
    pixel_coords = [
        (180, 173),
        (191, 164),
        (214, 165),
        (237, 167),
        (260, 169),
        (283, 170),
        (306, 172),
        (304, 195),
        (303, 218),
        (301, 241),
        (324, 243),
        (347, 244),
        (370, 246),
        (368, 269),
        (367, 292),
        (365, 315),
        (363, 338),
        (362, 361),
        (385, 363),
        (408, 364),
        (406, 387),
        (404, 410),
        (403, 433),
        (413, 438)
    ]

    for (u, v) in pixel_coords:
        move_robot_point(device ,M, u, v)

    device.close() 
    
if __name__ == "__main__":
    main()
