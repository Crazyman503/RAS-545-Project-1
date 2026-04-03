import numpy as np
import cv2
import time
import pydobot
from camera_utilities import apply_affine, fit_affine, apply_homography, fit_homography
from robot_utilities_2 import move_to_home, move_to_specific_position, get_current_pose

M = np.array(  [
    [ 8.27769622e-03, -4.92726589e-01, 4.48264448e+02],
    [-4.82986895e-01, -1.19049886e-04, 1.42131896e+02]]
, dtype=np.float64)

H = np.array( [
    [ 1.84671341e-02, -5.04797652e-01,  4.38827535e+02],
    [-4.60983168e-01, -1.17080254e-02 , 1.39659071e+02],
    [ 4.14897953e-05, -1.42704014e-04 , 1.00000000e+00]]
    , dtype=np.float64)

def move_robot_point(device,M,u,v):
    Xa, Ya = apply_affine(M, u, v) # Using Affine
    # Xa, Ya = apply_homography(H, u, v) # Using Homography
    print(f"Affine:  pixel({u:.3f}, {v:.3f}) -> robot({Xa:.6f}, {Ya:.6f})")
    move_to_specific_position(device, x=Xa, y=Ya, z=-45)
    time.sleep(1)

def main():
    device = pydobot.Dobot(port="COM5")
    device.speed(100, 100)
    move_to_home(device)
    time.sleep(2)

    # Example pixel coordinates from clicks
    pixel_coords = [
        (414, 235),
        (411, 241),
        (393, 259),
        (375, 276),
        (357, 294),
        (340, 276),
        (322, 258),
        (304, 275),
        (286, 293),
        (269, 275),
        (251, 257),
        (234, 239),
        (216, 257),
        (198, 274),
        (180, 256),
        (163, 238),
        (161, 232)
    ]

    for (u, v) in pixel_coords:
        move_robot_point(device ,M, u, v)

    device.close() 
    
if __name__ == "__main__":
    main()
