"""
reg_centerline.py
Robust SE(2) pose correction using LiDAR centroids (body frame) vs reference centerline.

- Uses current pose estimate (x,y,psi) to map centroids to map-frame.
- Residuals: signed lateral distances to nearest centerline points + heading residual vs local tangent.
- Huber loss + trimming to reject outliers.
- Continuity-constrained nearest using a window around an evolving anchor index.
- Levenberg–Marquardt damping + trust region to prevent divergence.
- Returns corrected pose (x+,y+,psi+) and an estimated 3x3 measurement covariance R3 for EKF.
"""

from __future__ import annotations
import numpy as np
from typing import Optional
from map_centerline import Centerline

def _wrap(a: float) -> float:
    return (a + np.pi) % (2*np.pi) - np.pi

def _huber_weights(r: np.ndarray, c: float) -> np.ndarray:
    a = np.abs(r)
    w = np.ones_like(r)
    m = a > c
    w[m] = c / (a[m] + 1e-9)
    return w

def _project_body_to_map(C_body: np.ndarray, x: float, y: float, psi: float) -> np.ndarray:
    c, s = np.cos(psi), np.sin(psi)
    R = np.array([[c, -s], [s, c]], dtype=float)
    return (C_body @ R.T) + np.array([x, y], dtype=float)

def _signed_lateral(p: np.ndarray, p_ref: np.ndarray, t_ref: np.ndarray):
    # lateral = n_ref^T (p - p_ref), where n_ref = [-t_y, t_x]
    n = np.stack([-t_ref[:, 1], t_ref[:, 0]], axis=1)
    d = p - p_ref
    return np.sum(n * d, axis=1), n  # (N,), (N,2)

def robust_pose_correction_centerline_win(
    C_body_xy: np.ndarray,
    cl: Centerline,
    last_idx: Optional[int],
    window_m: float,
    x0: float, y0: float, psi0: float,
    max_iters: int = 6,
    trim_frac: float = 0.35,
    huber_c_lat: float = 0.25,
    huber_c_psi: float = np.deg2rad(5.0)
):
    """
    Return (x+, y+, psi+, R3_est, new_last_idx).
    If C_body_xy is empty or system ill-posed, returns input pose and a safe R3.
    """
    if C_body_xy.shape[0] == 0:
        R3 = np.diag([1.0, 1.0, np.deg2rad(10.0)**2])
        return x0, y0, psi0, R3, (last_idx if last_idx is not None else 0)

    x, y, psi = float(x0), float(y0), float(psi0)
    # window in points (≥10)
    w_pts = max(10, int(window_m / max(1e-3, cl.mean_ds)))

    # Continuity anchor that evolves each iteration
    anchor_idx = last_idx

    I3 = np.eye(3)
    Jk = rk = None  # for covariance fallback

    # Precompute per-point *range* in the body frame for distance down-weighting
    r_body = np.hypot(C_body_xy[:, 0], C_body_xy[:, 1])
    # Smooth distance roll-off ~15 m
    def _w_dist():
        return np.exp(- (r_body / 15.0)**2)

    for _ in range(max_iters):
        # Map points with current pose
        Pm = _project_body_to_map(C_body_xy, x, y, psi)

        # Nearest lookup: global on first iteration, windowed thereafter
        if anchor_idx is None:
            dist, idx = cl.kdt.query(Pm, k=1)
            Pref = cl.xy[idx]
            Tref = cl.tangents[idx]
            anchor_idx = int(np.median(idx))
        else:
            idx, Pref, Tref = cl.nearest_window(Pm, anchor_idx, w_pts)

        # Residuals
        lat, n_ref = _signed_lateral(Pm, Pref, Tref)   # lateral distance
        psi_ref = np.arctan2(Tref[:, 1], Tref[:, 0])   # local tangent heading
        dpsi = _wrap(np.full_like(lat, psi) - psi_ref) # heading residual

        Np = len(lat)
        if Np < 4:
            break

        # Jacobians
        lever = np.stack([Pm[:, 0] - Pref[:, 0], Pm[:, 1] - Pref[:, 1]], axis=1)
        dlat_dpsi = -n_ref[:, 0] * lever[:, 1] + n_ref[:, 1] * lever[:, 0]
        J_lat = np.column_stack([n_ref, dlat_dpsi])

        J_psi = np.zeros((Np, 3), dtype=float); J_psi[:, 2] = 1.0

        # Stack system
        J = np.zeros((2 * Np, 3), dtype=float)
        r = np.zeros(2 * Np, dtype=float)
        J[:Np, :] = J_lat;  r[:Np] = lat
        J[Np:, :] = J_psi;  r[Np:] = dpsi

        # Robust weights + additional priors:
        # - distance roll-off (prefer close features)
        # - alignment roll-off (prefer small heading mismatch)
        w_lat = _huber_weights(r[:Np], huber_c_lat)
        w_psi = _huber_weights(r[Np:], huber_c_psi)
        w_align = np.exp(- (np.abs(dpsi) / np.deg2rad(25.0))**2)
        w = np.concatenate([w_lat * _w_dist(), w_psi * w_align], axis=0)

        # Adaptive trimming (more aggressive when lots of points)
        trim = 0.5 if Np > 60 else trim_frac
        k = int((1.0 - trim) * len(w))
        if k < 3:
            break
        order = np.argsort(np.abs(w * r))
        keep = order[:k]
        W = np.diag(w[keep])
        Jk = J[keep, :]
        rk = r[keep]

        # Normal equations + LM damping
        H = Jk.T @ W @ Jk
        g = Jk.T @ W @ rk
        lam = 1e-3 * (np.trace(H) / 3.0 + 1e-9)  # scale-aware damping
        H_d = H + lam * I3

        try:
            delta = -np.linalg.solve(H_d, g)
        except np.linalg.LinAlgError:
            break

        # Trust-region clamp (per-iter safety)
        max_xy = 0.8              # m
        max_psi = np.deg2rad(8.0) # rad
        step_xy = float(np.hypot(delta[0], delta[1]))
        if step_xy > max_xy:
            s = max_xy / (step_xy + 1e-9)
            delta[0] *= s; delta[1] *= s
        delta[2] = float(np.clip(delta[2], -max_psi, +max_psi))

        # Apply step
        x += float(delta[0]); y += float(delta[1]); psi = _wrap(psi + float(delta[2]))

        # Update continuity anchor using median of matched indices
        anchor_idx = int(np.median(idx))

        # Converged?
        if np.linalg.norm(delta) < 1e-3:
            break

    # Covariance estimate from final normal matrix (robust scale)
    if Jk is not None and rk is not None:
        if 'delta' not in locals():
            delta = np.zeros(3, dtype=float)
        res = rk + Jk @ delta
        sigma2 = (1.4826 * np.median(np.abs(res)) + 1e-6)**2
        N = (Jk.T @ Jk) + 1e-9 * I3
        R3 = np.linalg.pinv(N) * sigma2

        # Ensure SPD + floors
        R3 = 0.5 * (R3 + R3.T)
        R3[0, 0] = max(R3[0, 0], 0.02**2)
        R3[1, 1] = max(R3[1, 1], 0.02**2)
        R3[2, 2] = max(R3[2, 2], np.deg2rad(1.0)**2)
        wvals, V = np.linalg.eigh(R3)
        wvals = np.clip(wvals, 1e-8, None)
        R3 = V @ np.diag(wvals) @ V.T
    else:
        R3 = np.diag([0.05**2, 0.05**2, np.deg2rad(2.0)**2])

    return float(x), float(y), float(psi), R3, (anchor_idx if anchor_idx is not None else 0)
