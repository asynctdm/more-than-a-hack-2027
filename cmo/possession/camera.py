"""Camera centre on the pitch, from the per-frame homography (pitch 0-1 -> pixels).

The tactical camera pans with play, so the pitch point under the image centre is a coarse,
always-available proxy for where the action is (median error ~12-16 m against the tracked ball).
"""

import numpy as np

IMAGE_CENTRE = (960.0, 540.0)   # 1920x1080 tactical video


def camera_centre(H, reprojection_error, valid):
    x = np.full(len(H), np.nan)
    y = np.full(len(H), np.nan)
    Hi = np.linalg.inv(H[valid])
    v = Hi @ np.array([IMAGE_CENTRE[0], IMAGE_CENTRE[1], 1.0])
    x[valid] = v[:, 0] / v[:, 2]
    y[valid] = v[:, 1] / v[:, 2]
    return x, y
