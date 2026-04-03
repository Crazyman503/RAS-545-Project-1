# filename: pixel_to_robot_mapper.py
import numpy as np
import cv2

# ================================
# 1) Put your calibration pairs here
#    Image (u,v)  ->  Robot (X,Y)
# ================================

# Clicked at: x=162, y=45
# Clicked at: x=358, y=51
# Clicked at: x=494, y=45
# Clicked at: x=168, y=179
# Clicked at: x=366, y=176
# Clicked at: x=428, y=175
# Clicked at: x=492, y=173

img_pts = np.array([
    [508, 425],
    [304, 420],
    [95, 419],
    [427, 381],
    [299, 379],
    [176, 377],
    [426, 260],
    [303, 259],
    [183, 257],
    [508, 222],
    [306, 220],
    [102, 214]
], dtype=np.float64)

rob_xy = np.array([
    [243.35379028320312, -103.48576354980469],
    [242.24429321289062, -2.891136646270752],
    [242.62620544433594, 96.2242202758789],
    [263.6775817871094, -64.03107452392578],
    [264.4687194824219, -3.1356115341186523],
    [263.9208679199219, 57.21224594116211],
    [320.98187255859375, -65.5381088256836],
    [323.17889404296875, -3.8888068199157715],
    [323.24029541015625, 53.29458236694336],
    [340.0483703613281, -105.75225067138672],
    [339.9085998535156, -3.9766266345977783],
    [341.5600280761719, 92.49080657958984]
], dtype=np.float64)

def fit_affine(img_pts, rob_xy):
    """Fit affine [X Y]^T = M * [u v 1]^T using OpenCV."""
    M, inliers = cv2.estimateAffine2D(
        img_pts.reshape(-1,1,2),
        rob_xy.reshape(-1,1,2),
        ransacReprojThreshold=1.0,
        refineIters=1000
    )
    if M is None:
        raise RuntimeError("Affine estimation failed. Points may be degenerate.")
    return M

def fit_homography(img_pts, rob_xy):
    """Fit projective H so that [X Y 1]^T ~ H * [u v 1]^T."""
    H, inliers = cv2.findHomography(img_pts, rob_xy, method=cv2.RANSAC, ransacReprojThreshold=1.0)
    if H is None:
        raise RuntimeError("Homography estimation failed. Points may be degenerate.")
    return H

def apply_affine(M, u, v):
    """Apply affine transform (2x3) to a single pixel (u,v) -> (X,Y)."""
    uv1 = np.array([u, v, 1.0], dtype=np.float64)
    XY = M @ uv1
    return float(XY[0]), float(XY[1])

def apply_homography(H, u, v):
    """Apply homography (3x3) to a single pixel (u,v) -> (X,Y)."""
    uv1 = np.array([u, v, 1.0], dtype=np.float64)
    Xp, Yp, W = H @ uv1
    if abs(W) < 1e-12:
        raise ZeroDivisionError("Homography scale ~ 0 for this point.")
    return float(Xp / W), float(Yp / W)

def rms_error_affine(M, img_pts, rob_xy):
    ones = np.ones((img_pts.shape[0], 1))
    uv1 = np.hstack([img_pts, ones])          # (N,3)
    pred = (uv1 @ M.T)                        # (N,2)
    err = rob_xy - pred
    return float(np.sqrt(np.mean(np.sum(err**2, axis=1))))

def rms_error_homography(H, img_pts, rob_xy):
    uv1 = np.hstack([img_pts, np.ones((img_pts.shape[0],1))])  # (N,3)
    proj = (uv1 @ H.T)                                         # (N,3)
    proj_xy = proj[:, :2] / proj[:, 2:3]
    err = rob_xy - proj_xy
    return float(np.sqrt(np.mean(np.sum(err**2, axis=1))))

def main():
    # --- Fit both models ---
    M = fit_affine(img_pts, rob_xy)
    H = fit_homography(img_pts, rob_xy)

    print("Affine matrix M (2x3):\n", M)
    print("\nHomography H (3x3):\n", H)

    # --- Report RMS fit error ---
    aff_rms = rms_error_affine(M, img_pts, rob_xy)
    hom_rms = rms_error_homography(H, img_pts, rob_xy)
    print(f"\nRMS error (affine):    {aff_rms:.6f} (robot units)")
    print(f"RMS error (homography): {hom_rms:.6f} (robot units)")
if __name__ == "__main__":
    main()
