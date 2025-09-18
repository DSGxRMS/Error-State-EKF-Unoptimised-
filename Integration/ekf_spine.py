"""
ekf_spine.py
Minimal error-state EKF for vehicle odometry and online noise adaptation.

What this module is (short):
 - An error-state Extended Kalman Filter tuned for 2D vehicle odometry.
 - Maintains a compact 6D state: [x, y, psi, v, gyro_bias_z, accel_bias_x].
 - Predicts at IMU rate (wz, ax) and accepts lightweight updates:
     * wheel/speed updates (treated like an encoder)
     * low-rate pose corrections (map/registration)
     * yaw-only corrections (fast heading fixes)
     * optional ZUPT when stopped
 - Includes simple online adaptation:
     * adapt wheel-measurement noise (Rv) using robust matching
     * gently inflate/process noise (Q) when residuals indicate under-modeling
     * small, bounded reactions to large but plausible residuals

Why use this design (short, non-mathy):
 - Compact and fast: a 6-state filter keeps CPU low and is easy to reason about.
 - Practical for robotics/autonomy where IMU+wheel odometry are primary and
   occasional map/scan corrections are available.
 - Online adaptation reduces the need to hand-tune measurement noise for each run
   and helps the filter remain useful across sessions with slightly different sensors.
 - Separate yaw-only updates let you correct heading quickly from a trusted source
   without forcing large X/Y jumps.
 - ZUPT is included to stabilise behavior around standstill (useful in repeated start/stop traffic).
 - Numerically defensive: guards for matrix inversion and avoids catastrophic failures.

This file intentionally focuses on practical behaviour and stability rather than
formal derivations. Keep it simple and robust by design.
"""

from __future__ import annotations
from dataclasses import dataclass
from collections import deque
import math
import numpy as np
from typing import Optional, Deque, Tuple

def _wrap(a: float) -> float:
    """Wrap angle to (-pi, pi]."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi

@dataclass
class SpineConfig:
    # Initial covariance guesses — keep these as practical starting points.
    P0_x: float = 1.0**2
    P0_y: float = 1.0**2
    P0_psi: float = (5.0 * math.pi / 180.0)**2  # ~5°
    P0_v: float = 1.0**2
    P0_bg: float = (0.05)**2                    # gyro-bias
    P0_ba: float = (0.5)**2                     # accel-bias

    # Base process noise levels (per-second). These are floors you can adapt at runtime.
    q_psi: float = (0.01)**2
    q_v: float   = (0.3)**2
    q_bg: float  = (0.002)**2
    q_ba: float  = (0.02)**2

    # Hard floors for process noise to avoid collapse.
    q_floor_psi: float = (0.005)**2
    q_floor_v: float   = (0.15)**2
    q_floor_bg: float  = (0.001)**2
    q_floor_ba: float  = (0.01)**2

    # Speed measurement noise (wheel/encoder) baseline and floor.
    Rv0: float      = (0.20)**2
    Rv_floor: float = (0.10)**2

    # Forgetting factor for R adaptation (near 1 => slow change)
    beta_R: float = 0.997

    # Simple NIS window + thresholds for gentle Q scaling when needed.
    nis_win: int = 200
    nis_hi_ratio: float = 1.30
    q_inflate: float = 1.15

    # Only adapt while moving (prevents garbage adaptation while parked)
    adapt_motion_gate_mps: float = 0.30

    # ZUPT thresholds (small, practical values)
    zupt_speed_thresh: float = 0.15   # m/s
    zupt_hold_sec: float    = 0.25
    zupt_R: float           = (0.05)**2

class ESEKFSpine:
    """
    Error-state EKF wrapper providing:
      - predict(dt, wz_imu, ax_imu)
      - update_speed(v_meas, age_sec)
      - update_pose2d(x,y,psi,R3)
      - update_yaw(psi,Rpsi)
      - zupt_if_needed(dt, speed_meas)
    Persistence and monitoring is left to the caller.
    """
    def __init__(self, cfg: SpineConfig = SpineConfig()):
        self.cfg = cfg
        # State: x = [x, y, psi, v, bg, ba]
        self.x = np.zeros(6, dtype=float)
        # Covariance initialised from config
        self.P = np.diag([cfg.P0_x, cfg.P0_y, cfg.P0_psi, cfg.P0_v, cfg.P0_bg, cfg.P0_ba])
        # Base process-noise that may be gently adjusted online
        self._q_base = np.array([cfg.q_psi, cfg.q_v, cfg.q_bg, cfg.q_ba], dtype=float)
        # Wheel-speed measurement variance (learned/updated online)
        self.Rv = float(cfg.Rv0)
        # NIS rolling window used for light Q adaptation
        self._nis_win: Deque[float] = deque(maxlen=cfg.nis_win)
        # ZUPT accumulator
        self._zupt_accum_t = 0.0

        # Time since motion started — used to be conservative immediately after moving
        self._moving_time = 0.0

    # ---------- Predict ----------
    def predict(self, dt: float, wz_imu: float, ax_imu: float) -> None:
        """
        Propagate the state forward by dt using IMU inputs.
        Keeps track of a simple motion timer to gate online adaptation.
        """
        if dt <= 0.0:
            return
        x, y, psi, v, bg, ba = self.x
        wz = wz_imu - bg   # remove estimated gyro bias
        ax = ax_imu - ba   # remove estimated accel bias

        # Basic vehicle kinematic update (forward integration)
        x  += v * math.cos(psi) * dt
        y  += v * math.sin(psi) * dt
        psi = _wrap(psi + wz * dt)
        v  += ax * dt
        self.x[:] = [x, y, psi, v, bg, ba]

        # State-transition linearisation stored in F
        F = np.eye(6)
        F[0,2] = -v * math.sin(psi) * dt
        F[0,3] =  math.cos(psi) * dt
        F[1,2] =  v * math.cos(psi) * dt
        F[1,3] =  math.sin(psi) * dt
        F[2,4] = -dt
        F[3,5] = -dt

        # Process noise: ensure no values below configured floors
        q = np.maximum(self._q_base, np.array([
            self.cfg.q_floor_psi, self.cfg.q_floor_v, self.cfg.q_floor_bg, self.cfg.q_floor_ba
        ]))
        Qd = np.zeros((6,6), dtype=float)
        Qd[2,2] = q[0] * dt
        Qd[3,3] = q[1] * dt
        Qd[4,4] = q[2] * dt
        Qd[5,5] = q[3] * dt

        # Covariance propagate
        self.P = F @ self.P @ F.T + Qd

        # Update moving-time tracker used to avoid aggressive adaptation when stationary
        if abs(self.x[3]) > self.cfg.adapt_motion_gate_mps:
            self._moving_time += dt
        else:
            self._moving_time = 0.0

    # ---------- Speed update (CarState.speed) ----------
    def update_speed(self, v_meas: float, age_sec: float = 0.0) -> Tuple[float, float]:
        """
        Fuse a wheel-speed measurement.
        Returns (innovation, NIS).
        Includes guarded behaviour early after motion and per-update inflation
        to avoid transient large errors destabilizing baselines.
        """
        H = np.zeros((1,6), dtype=float); H[0,3] = 1.0
        z = np.array([v_meas], dtype=float)

        # Contribution of prior covariance to measurement
        HPH = float(self.P[3,3])

        # If measurement is stale, down-weight it slightly (simple age scaling)
        age_scale = (1.0 + 0.5 * max(0.0, age_sec))
        R_eff = max(self.cfg.Rv_floor, float(self.Rv) * age_scale)

        S = H @ self.P @ H.T + R_eff
        y = float(z - H @ self.x)

        # Normalized residual magnitude used for simple gates
        sigma_pred = math.sqrt(max(1e-12, HPH + max(self.cfg.Rv_floor, self.Rv)))
        r_norm = abs(y) / sigma_pred

        # Early-motion hard gate: skip wildly large residuals right after motion starts
        if (self._moving_time < 1.5) and (r_norm > 8.0):
            nis = float((y * y) / float(S))
            return y, nis

        # Soft per-update inflation for large but plausible residuals
        if r_norm > 3.0:
            scale = min(4.0, r_norm / 3.0)
            R_eff = max(R_eff, R_eff * scale)
            S = H @ self.P @ H.T + R_eff

        # Kalman update for speed
        K = (self.P @ H.T) / float(S)
        self.x = self.x + (K.flatten() * y)
        self.x[2] = _wrap(self.x[2])
        if self.x[3] < 0.0:
            # Enforce non-negative forward speed
            self.x[3] = 0.0

        I = np.eye(6)
        # Joseph form for covariance update (numerically safer)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K * R_eff * K.T

        nis = float((y * y) / float(S))

        # --------- Motion-gated adaptation ---------
        moving = (abs(v_meas) > self.cfg.adapt_motion_gate_mps) or (abs(self.x[3]) > self.cfg.adapt_motion_gate_mps)
        if moving:
            # collect NIS for Q-scaling if residuals aren't extreme
            if r_norm <= 8.0:
                self._nis_win.append(nis)

            # Robust R estimation from sample residuals (bounded)
            est_R = max(self.cfg.Rv_floor, y*y - HPH)

            # If residuals are large, route some effect into process noise as well
            if r_norm > 4.0:
                est_R = min(est_R, self.Rv * 3.0)
                self._q_base[1] = min(self._q_base[1] * 1.10, 5.0)

            # Exponential forgetting (Sage–Husa style) for Rv
            beta = self.cfg.beta_R
            self.Rv = max(self.cfg.Rv_floor, beta * float(self.Rv) + (1.0 - beta) * est_R)

            # Gentle Q scaling based on NIS statistics (inflation only)
            self._maybe_scale_Q()

        return y, nis

    # ---------- Pose correction (x,y,psi) ----------
    def update_pose2d(self, x_m: float, y_m: float, psi_m: float, R3: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Apply a 3D pose correction (x, y, psi) with covariance R3.
        Returns (innovation vector, NIS).
        Uses safe inversion and Joseph-form covariance update.
        """
        z = np.array([x_m, y_m, _wrap(psi_m)], dtype=float)
        H = np.zeros((3,6), dtype=float)
        H[0,0] = 1.0; H[1,1] = 1.0; H[2,2] = 1.0

        xhat = self.x.copy()
        xhat[2] = _wrap(xhat[2])

        # Wrapped innovation for heading
        y = z - H @ xhat
        y[2] = _wrap(y[2])

        R = 0.5*(R3 + R3.T)
        S = H @ self.P @ H.T + R

        # Inversion guard: add tiny jitter until invertible, otherwise pseudo-inverse
        I3 = np.eye(3); jitter = 1e-9
        for _ in range(6):
            try:
                Sinv = np.linalg.inv(S)
                break
            except np.linalg.LinAlgError:
                S = S + jitter * I3
                jitter *= 10.0
        else:
            Sinv = np.linalg.pinv(S)

        K = self.P @ H.T @ Sinv
        self.x = self.x + K @ y
        self.x[2] = _wrap(self.x[2])

        I = np.eye(6)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R @ K.T

        nis = float(y.T @ Sinv @ y)
        return y, nis

    # ---------- Yaw-only correction ----------
    def update_yaw(self, psi_meas: float, Rpsi: float) -> Tuple[float, float]:
        """
        Fuse a single yaw measurement. Returns (innovation, NIS).
        Useful for quick heading fixes when a reliable yaw source is available.
        """
        psi_meas = _wrap(psi_meas)
        H = np.zeros((1,6), dtype=float); H[0,2] = 1.0
        z = np.array([psi_meas], dtype=float)

        # innovation with wrap
        y = float(_wrap(z[0] - self.x[2]))

        S = float(self.P[2,2] + max(1e-6, Rpsi))
        K = (self.P @ H.T) / S

        self.x = self.x + (K.flatten() * y)
        self.x[2] = _wrap(self.x[2])

        I = np.eye(6)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K * Rpsi * K.T

        nis = float((y*y) / S)
        return y, nis

    # ---------- ZUPT ----------
    def zupt_if_needed(self, dt: float, speed_meas: float) -> Optional[Tuple[float,float]]:
        """
        If the vehicle has been effectively stationary for a short hold time, apply
        a zero-velocity update to clamp speed and reduce drift.
        Returns (innovation, NIS) if applied, otherwise None.
        """
        if abs(speed_meas) < self.cfg.zupt_speed_thresh:
            self._zupt_accum_t += dt
        else:
            self._zupt_accum_t = 0.0
        if self._zupt_accum_t < self.cfg.zupt_hold_sec:
            return None

        H = np.zeros((1,6), dtype=float); H[0,3] = 1.0
        Rz = self.cfg.zupt_R
        S = H @ self.P @ H.T + Rz
        y = float(0.0 - H @ self.x)
        K = (self.P @ H.T) / float(S)
        self.x = self.x + (K.flatten() * y)
        self.x[2] = _wrap(self.x[2])
        if self.x[3] < 0.0:
            self.x[3] = 0.0

        I = np.eye(6)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K * Rz * K.T
        nis = float((y * y) / float(S))
        return (y, nis)

    # ---------- Helpers ----------
    def _maybe_scale_Q(self) -> None:
        """
        Light Q scaling based on NIS statistics: if the average NIS is persistently
        high, inflate the base process noise slightly. Floors enforced.
        """
        if len(self._nis_win) < self.cfg.nis_win // 2:
            return
        nis_avg = float(np.mean(self._nis_win))
        chi2_95 = 3.841458820694124  # reference for 1 DOF
        ratio = nis_avg / chi2_95
        if ratio > self.cfg.nis_hi_ratio:
            self._q_base *= self.cfg.q_inflate
        self._q_base = np.maximum(
            self._q_base,
            np.array([self.cfg.q_floor_psi, self.cfg.q_floor_v, self.cfg.q_floor_bg, self.cfg.q_floor_ba])
        )

    # ---------- Accessors ----------
    def state(self) -> Tuple[float,float,float,float,float,float]:
        """Return a copy of the current state (float tuple)."""
        return tuple(map(float, self.x))

    def covariance(self) -> np.ndarray:
        """Return a copy of the current covariance matrix."""
        return self.P.copy()
