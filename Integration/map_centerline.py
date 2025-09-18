"""
map_centerline.py
Builds a dense centerline from PATHPOINTS_CSV and exposes nearest-point queries,
including a continuity-constrained (windowed) nearest around a running index.
"""

import numpy as np
import pandas as pd
from scipy.interpolate import splprep, splev
from scipy.spatial import cKDTree

class Centerline:
    def __init__(self, xy: np.ndarray, s: np.ndarray, tangents: np.ndarray, kdt: cKDTree, is_loop: bool, mean_ds: float):
        self.xy = xy            # (N,2)
        self.s = s              # arclength samples [m]
        self.tangents = tangents  # unit (N,2)
        self.kdt = kdt
        self.is_loop = is_loop
        self.mean_ds = float(mean_ds)

    def nearest_window(self, pts_xy: np.ndarray, idx_center: int, w_pts: int):
        """
        Windowed nearest around idx_center ± w_pts (clamped if non-loop).
        Returns (idx_global, points_on_cl, tangents_on_cl).
        """
        N = self.xy.shape[0]
        if self.is_loop:
            idxs = (np.arange(idx_center - w_pts, idx_center + w_pts + 1) % N)
            slice_xy = self.xy[idxs]
            # brute-force distances in the window
            d2 = ((pts_xy[:, None, :] - slice_xy[None, :, :])**2).sum(axis=2)  # (M, W)
            j_local = np.argmin(d2, axis=1)
            idx_global = idxs[j_local]
        else:
            i0 = max(0, idx_center - w_pts)
            i1 = min(N, idx_center + w_pts + 1)
            slice_xy = self.xy[i0:i1]
            d2 = ((pts_xy[:, None, :] - slice_xy[None, :, :])**2).sum(axis=2)
            j_local = np.argmin(d2, axis=1)
            idx_global = i0 + j_local
        return idx_global, self.xy[idx_global], self.tangents[idx_global]


def _resample_track(x_raw, y_raw, num=3000):
    tck, _ = splprep([x_raw, y_raw], s=0, k=min(3, max(1, len(x_raw)-1)))
    tt = np.linspace(0,1,num)
    xx, yy = splev(tt, tck)
    return np.asarray(xx), np.asarray(yy)

def build_centerline(path_csv: str, transform=True, is_loop=False) -> Centerline:
    df = pd.read_csv(path_csv)
    x_raw = df["x"].to_numpy()
    y_raw = df["y"].to_numpy()

    # Resample and apply the same transform your controller uses
    rx, ry = _resample_track(x_raw, y_raw, num=3000)
    if transform:
        cx = ry + 15.0
        cy = -rx
    else:
        cx, cy = rx, ry

    # Arc-length & uniform re-sample at ~0.20 m spacing
    dx = np.gradient(cx); dy = np.gradient(cy)
    ds = np.hypot(dx, dy)
    s = np.concatenate(([0.0], np.cumsum(ds[1:])))
    L = s[-1] if s[-1] > 0 else 1.0
    target = np.linspace(0, L, int(max(50, L/0.20)))
    ux = np.interp(target, s, cx)
    uy = np.interp(target, s, cy)

    # Tangents & KD-tree
    dux = np.gradient(ux); duy = np.gradient(uy)
    nrm = np.hypot(dux, duy); nrm[nrm==0] = 1.0
    tx = dux / nrm; ty = duy / nrm
    xy = np.stack([ux, uy], axis=1)
    tangents = np.stack([tx, ty], axis=1)
    kdt = cKDTree(xy)

    mean_ds = float(np.mean(np.hypot(np.diff(ux), np.diff(uy))))
    return Centerline(xy, target, tangents, kdt, bool(is_loop), mean_ds)
