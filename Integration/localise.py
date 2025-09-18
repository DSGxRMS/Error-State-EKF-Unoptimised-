# localise.py 
# -----------------------------------------------------------------------------
# Dependencies (other project files / modules this script relies on)
#
#   ekf_spine.py            -> ESEKFSpine, SpineConfig
#   sensors_fsds.py         -> FSDSReaders, FSDSConfig (abstracted sensor readers)
#   lidar_frontend_basic.py -> LidarFrontend (centroid extraction / lidar preproc)
#   map_centerline.py       -> build_centerline (path centerline loader)
#   reg_centerline.py       -> robust_pose_correction_centerline_win (registration)
#   learned_params.py       -> LearnedParams (persisted learned baselines)
#
# Optional:
#   fsds (python package)   -> optional runtime ground-truth client (FSDS simulator)
#
# Purpose (short):
#   Real-time localisation supervisor that:
#     - Runs an error-state EKF (ESEKFSpine) using IMU+wheels + LiDAR→centerline
#     - Applies periodic LiDAR→centerline pose corrections with NIS gating
#     - Optionally uses FSDS ground-truth for "assist" or "strict" supervision
#     - Publishes a smoothed, fixed-rate controller state stream (ControllerBus)
#     - Implements a strong yaw-correction leash using GT yaw to quickly fix drift
#     - Learns & persists simple calibration baselines (Rv, gyro_bias, v_scale)
#
# Design notes (why these pieces):
#   - EKF approach: lightweight ESEKF keeps a compact 6D vehicle state and supports
#     measurement updates (speed, pose2D) and covariance handling needed for NIS gating.
#   - LiDAR→centerline registration: robust matching of extracted centroids to a
#     precomputed centerline provides relative pose observations without a full map.
#   - NIS checks: predicted NIS calculation prevents overconfident or inconsistent
#     registration updates; separate "yaw-only" gating permits coarse yaw corrections
#     even when positional evidence is weak.
#   - Controller bus: many controllers expect a fixed-rate, low-latency state feed.
#     We buffer irregular EKF outputs, resample into a stable sim-time grid, apply
#     latency-forward prediction, exponential smoothing, and publish with rate limits.
#   - GT supervision modes: 'off'|'assist'|'strict' let you choose how much to trust
#     simulator GT. Strict locks yaw and speed; assist nudges softly. Useful for tests.
#   - Strong yaw correction: a per-tick yaw-only correction (very large XY variance)
#     is used to "pull" EKF yaw to GT yaw quickly while avoiding teleporting X/Y.
#   - Persistence: learned baselines (Rv, gyro_bias, v_scale) are low-dimensional and
#     worth saving between runs — they stabilise behaviour across sessions.
#
# Tradeoffs and caution:
#   - Aggressive GT yaw locking will hide real estimator failures; use 'assist' for
#     development unless you're debugging specific yaw issues.
#   - Controller smoothing and rate limits intentionally damp actuations; tune to
#     controller dynamics and actuation latency for best results.
# -----------------------------------------------------------------------------

from __future__ import annotations
import os, csv, time, math, json, socket, hashlib
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import deque

from ekf_spine import ESEKFSpine, SpineConfig
from sensors_fsds import FSDSReaders, FSDSConfig
from lidar_frontend_basic import LidarFrontend
from map_centerline import build_centerline
from reg_centerline import robust_pose_correction_centerline_win

# Optional ground truth (FSDS) import - gracefully degrade if not available.
try:
    import fsds
    _FSDS_OK = True
except Exception:
    _FSDS_OK = False

# Persistent learned params (simple single-file JSON store)
from learned_params import LearnedParams
LP_PATH = "./logs/learned_params.json"
LP = LearnedParams.load(LP_PATH)

# ================= GLOBAL SWITCHES =================
# High-level, human-friendly knobs for experiment workflows.
GT_MODE = "strict"          # "off" | "assist" | "strict" — controls how much GT to apply
PERSIST_LEARNING = True     # periodically save learned baselines back to disk

# -------- Legacy stream (unchanged) --------
# Backwards-compatible UDP stream for other tooling that expects irregular predictions.
STREAM_ENABLE = True
STREAM_HOST   = "127.0.0.1"
STREAM_PORT   = 5601
STREAM_HZ     = 30.0

# -------- Controller bus (fixed-rate) --------
# Provides a stable, smoothed, latency-compensated state feed for downstream
# controllers. It buffers irregular incoming EKF samples, interpolates in time,
# predicts forward by a small actuation latency, smooths, and publishes at a
# fixed Hz. This prevents controllers from seeing jittery timestamps or large steps.
CTRL_STREAM_ENABLE   = True
CTRL_STREAM_HOST     = "127.0.0.1"
CTRL_STREAM_PORT     = 5602      # port controllers subscribe to
CTRL_STREAM_HZ       = 100.0     # publish rate (controller loop rate)
CTRL_LATENCY_S       = 0.08      # forward prediction horizon -> actuation command time
CTRL_STATE_BUFFER_S  = 0.35      # buffer depth used for interpolation
# Smoothing (1st-order low-pass): alpha = dt/(tau+dt)
CTRL_TAU_V_S         = 0.25      # smoothing time constant for speed (gentle low-pass)
CTRL_TAU_YAW_S       = 0.18      # smoothing time constant for yaw (angle-space EMA)
# Output rate limits to avoid step changes that shock the controller
CTRL_MAX_DPS         = 120.0     # deg/s cap for yaw rate of change published
CTRL_MAX_ACC         = 5.0       # m/s^2 cap on published longitudinal accel

# ---------------- Config ----------------
PATHPOINTS_CSV = "./data/vd_pathpoints.csv"
LOG_DIR   = "./logs"
LOG_FILE  = os.path.join(LOG_DIR, "stepE_supervised.csv")

# Registration cadence & window
REG_DT     = 0.12           # how often we attempt LiDAR->centerline registration
WINDOW_M   = 10.0           # search radius for registration
CENT_MIN   = 8              # min centroids required to consider registration
CENT_MAX   = 120            # max centroids (subsample down if more)

# NIS gates (tighter thresholds to reduce bad updates)
FULL2YAW_NIS = 28.0
YAW2SKIP_NIS = 95.0

# Innovation clamps & smoothing used after registration to avoid teleports
INNOV_CLAMP_POS = 0.50                 # max allowed position correction (m)
INNOV_CLAMP_YAW = np.deg2rad(6.0)      # max allowed yaw correction (rad)
YAW_EMA_ALPHA_REG = 0.30               # EMA alpha used when applying yaw innovation

# Curvature-aware trust: increase positional covariance on sharp corners
CURV_SPAN   = 6
CURV_STRONG = 0.020                    # 1/m threshold considered "strong" curvature

# s-progress gate: prevents registration from "teleporting" the s-progress
S_GATE_MARGIN_M   = 0.8
S_GATE_RATE_SCALE = 1.5

# Plotting / UI
PLOT_HZ        = 12.0
CAR_HEADING_M  = 2.0
TRAJ_BUFFER_N  = 20000
CAR_VIEW_HALF  = 22.0
USE_CAR_CENTER = True
PAUSE_PLOT     = False

# ---------------- GT knobs (more granular control) ----------------
STRICT_GT_YAW_LOCK   = False
STRICT_GT_SPEED_LOCK = False
STRICT_GT_SOFT_POS   = False
USE_GT_INIT          = False
USE_GT_LEARN_VSCALE  = False
USE_GT_GYRO_BIAS     = False
USE_GT_ADAPT_RV      = True           # adapt wheel Rv baseline from EKF NIS

GT_YAW_SIG_DEG   = 2.0
GT_POS_SOFT_M    = 0.35
GT_NUDGE_DT      = 0.35
GT_POS_R_XY      = 0.20
GT_POS_R_YAW_DEG = 4.0

# Speed/NIS adaptation parameters
NISV_TARGET = 10.0
NISV_ALPHA  = 0.05
RV_CLAMP    = (0.02, 0.40)

# v-scale learning (wheel speed scale factor)
VSCALE_ALPHA = 0.06
VSCALE_CLAMP = (0.88, 1.12)

# Persist learned baselines cadence
PERSIST_EVERY_S = 3.0

# ========= STRONG GT YAW CORRECTION (rationale + knobs) =========
# This block applies yaw-only corrections every IMU tick using a very large XY
# covariance so the EKF only uses the yaw component. Rationale:
#   - Yaw drift from IMU integration is a common, fast-growing error.
#   - Using GT yaw (when available) keeps heading aligned without teleporting XY.
#   - We use different gains/limits on straights vs corners to avoid fighting
#     valid path curvature-induced yaw differences.
YAW_DEADBAND_DEG      = 0.20
YAW_STEP_CAP_DEG      = 0.90     # per-tick cap (deg)
YAW_EMA_ALPHA_STRONG  = 0.55     # heavier EMA (faster locking compared to registration EMA)
YAW_SIG_STRAIGHT_DEG  = 0.60
YAW_SIG_CORNER_DEG    = 1.60
YAW_CURV_WEAK         = 0.008
YAW_CURV_STRONG       = 0.020
YAW_MICRO_ITERS_MAX   = 2        # extra micro-steps if residual still large on straights
# Startup "strict yaw" window: quick lock-in early in run
YAW_STRICT_WINDOW_S   = 3.0
YAW_SIG_STRICT_DEG    = 0.35
YAW_EMA_STRICT        = 0.70
YAW_STEP_STRICT_DEG   = 1.20
# Gyro-bias learning schedule: faster at run start, then taper
GYRO_BIAS_FAST_S      = 8.0      # seconds of faster learning
GYRO_BIAS_BETA_FAST   = 0.010    # per tick during fast window
GYRO_BIAS_BETA_SLOW   = 0.002    # per tick after fast window

# ========= helpers =========
def _wrap(a: float) -> float:
    # Normalize angle to (-π, π]
    return (a + math.pi) % (2.0 * math.pi) - math.pi

def _deg(v) -> float: return float(np.degrees(v))
def _rad(v) -> float: return float(np.radians(v))

def _interp01(x, x0, x1):
    # Linear interpolation fraction, clipped to [0,1]
    if x <= x0: return 0.0
    if x >= x1: return 1.0
    return float((x - x0) / max(1e-9, (x1 - x0)))

def _yaw_only_R3(sigma_deg: float) -> np.ndarray:
    # Build measurement covariance that effectively says: x,y are unknown (huge var),
    # only yaw has a finite variance. Used for yaw-only pose updates from GT.
    rpsi = _rad(max(1e-6, sigma_deg))
    return np.diag([1e6, 1e6, rpsi*rpsi])

def _ensure_logs():
    # Ensure log directory exists and create CSV header if missing.
    os.makedirs(LOG_DIR, exist_ok=True)
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "t","x","y","psi","v","nis_speed","Rv",
                "reg_mode","regNIS","num_centroids",
                "e_lat","e_yaw","s",
                "gt_x","gt_y","gt_psi",
                "dt_used","dup_flag","reg_ms",
                "rv_adapt","b_wz","v_scale","Rv_base","b_wz_base",
                "yaw_err_deg","ctrl_pub_t","ctrl_x","ctrl_y","ctrl_psi","ctrl_v"
            ])

def _predicted_nis_pose3(ekfP: np.ndarray, yv: np.ndarray, R3: np.ndarray) -> float:
    """
    Compute predicted NIS for a 3D pose innovation yv (x,y,psi).
    Uses measurement Jacobian H = [I3 | 0] on the EKF 6x6 covariance.
    Defensive: attempts solve, falls back to pseudo-inverse on failure.
    """
    H = np.zeros((3,6), dtype=float); H[0,0]=H[1,1]=H[2,2]=1.0
    S = H @ ekfP @ H.T + 0.5*(R3 + R3.T)
    I3 = np.eye(3); jitter = 1e-9
    for _ in range(6):
        try:
            Sinv = np.linalg.solve(S, I3); break
        except np.linalg.LinAlgError:
            # Add tiny diagonal until invertible
            S = S + jitter * I3; jitter *= 10.0
    else:
        Sinv = np.linalg.pinv(S)
    return float(yv.T @ Sinv @ yv)

def _nearest_idx_window(cl, x: float, y: float, last_idx, window_m: float):
    """
    Efficiently find centerline index nearest to (x,y) but search in a
    small window around last_idx if provided. This keeps registration
    anchored and avoids searching the whole centerline every time.
    """
    XY = cl.xy; N = XY.shape[0]
    if last_idx is None:
        d2 = (XY[:,0]-x)**2 + (XY[:,1]-y)**2
        return int(np.argmin(d2))
    # estimate ds spacing to convert meters -> indices
    ds_mean = float((cl.s[-1] - cl.s[0]) / max(1, N-1))
    win_pts = max(5, int(window_m / max(1e-3, ds_mean)))
    i0 = max(0, int(last_idx) - win_pts)
    i1 = min(N, int(last_idx) + win_pts + 1)
    sub = XY[i0:i1]
    d2 = (sub[:,0]-x)**2 + (sub[:,1]-y)**2
    return int(i0 + int(np.argmin(d2)))

def _signed_lateral_error(cl, x: float, y: float, psi: float, idx_guess=None, window_m=8.0):
    """
    Compute signed lateral error and heading difference relative to the centerline:
      - e_lat: signed lateral offset (positive = right of path normal convention)
      - e_yaw: yaw error relative to centerline tangent
      - s_here: arc-length along centerline at nearest index
      - idx: index of nearest centerline point
    """
    idx = _nearest_idx_window(cl, x, y, idx_guess, window_m)
    pt  = cl.xy[idx]
    tan = cl.tangents[idx]
    n   = np.array([-tan[1], tan[0]], dtype=float)
    v   = np.array([x - pt[0], y - pt[1]], dtype=float)
    e_lat = float(np.dot(v, n))
    psi_ref = math.atan2(tan[1], tan[0])
    e_yaw   = _wrap(psi - psi_ref)
    s_here  = float(cl.s[idx])
    return e_lat, e_yaw, s_here, idx

def _s_at_xy(cl, x: float, y: float, idx_hint=None, window_m=6.0) -> float:
    idx = _nearest_idx_window(cl, x, y, idx_hint, window_m)
    return float(cl.s[idx]), idx

def _quat_to_R3(qw, qx, qy, qz):
    # Robust quaternion->rotation (3x3) conversion with normalization guard.
    n2 = qw*qw + qx*qx + qy*qy + qz*qz
    if not np.isfinite(n2) or n2 <= 1e-12:
        return np.eye(3, dtype=float)
    s = 1.0 / math.sqrt(n2)
    qw, qx, qy, qz = qw*s, qx*s, qy*s, qz*s
    xx, yy, zz = qx*qx, qy*qy, qz*qz
    xy, xz, yz = qx*qy, qx*qz, qy*qz
    wx, wy, wz = qw*qx, qw*qy, qw*qz
    return np.array([
        [1 - 2*(yy + zz), 2*(xy - wz),     2*(xz + wy)],
        [2*(xy + wz),     1 - 2*(xx + zz), 2*(yz - wx)],
        [2*(xz - wy),     2*(yz + wx),     1 - 2*(xx + yy)]
    ], dtype=float)

def _yaw_from_quat(q):
    # Extract yaw from a quaternion using the rotation matrix
    R = _quat_to_R3(q.w_val, q.x_val, q.y_val, q.z_val)
    return math.atan2(R[1, 0], R[0, 0])

def _maybe_flip_centerline_direction_by_gt(cl, x0, y0, psi0):
    """
    On initialisation, flip centerline direction if the GT yaw indicates
    the path direction is reversed. Useful after spawn to ensure s increases
    along the vehicle motion direction.
    """
    idx0 = _nearest_idx_window(cl, x0, y0, None, 30.0)
    t = cl.tangents[idx0]
    psi_ref = math.atan2(t[1], t[0])
    d = abs(_wrap(psi0 - psi_ref))
    if d > math.radians(90.0):
        try:
            # reverse arrays and remap s to keep continuity
            cl.xy = cl.xy[::-1].copy()
            cl.tangents = cl.tangents[::-1].copy()
            s0 = cl.s[-1]
            cl.s = (s0 - cl.s[::-1]).copy()
            print("[StepE] Centerline direction flipped to match spawn heading.")
        except Exception:
            print("[StepE] WARN: centerline flip failed; continuing as-is.")

def _curvature_at_idx(cl, idx: int, span: int = CURV_SPAN) -> float:
    # Estimate curvature by finite-angle change over a span (|Δθ| / Δs).
    N = cl.xy.shape[0]
    if N < 3: return 0.0
    i0 = max(0, int(idx) - span)
    i1 = min(N - 1, int(idx) + span)
    th0 = math.atan2(cl.tangents[i0,1], cl.tangents[i0,0])
    th1 = math.atan2(cl.tangents[i1,1], cl.tangents[i1,0])
    dth = abs(_wrap(th1 - th0))
    ds  = max(1e-3, abs(cl.s[i1] - cl.s[i0]))
    return float(dth / ds)

def _clamp(v, vmin, vmax):
    # Simple clamp helper
    return vmin if v < vmin else (vmax if v > vmax else v)

def _limit_pose_innovation(x, y, psi, x_corr, y_corr, psi_corr,
                           yaw_cap=INNOV_CLAMP_YAW, pos_cap=INNOV_CLAMP_POS,
                           yaw_alpha=YAW_EMA_ALPHA_REG):
    """
    Limit raw registration corrections to avoid sudden teleports.
    - positions are scaled down if magnitude > pos_cap
    - yaw is clamped to yaw_cap and blended using yaw_alpha (EMA-like)
    Returns an adjusted (x', y', psi') safe to apply to the EKF.
    """
    dx  = float(x_corr - x)
    dy  = float(y_corr - y)
    dpsi= _wrap(float(psi_corr - psi))
    r = math.hypot(dx, dy)
    if r > pos_cap:
        sc = pos_cap / r
        dx *= sc; dy *= sc
    dpsi = _clamp(dpsi, -yaw_cap, yaw_cap)
    mpsi = _wrap(psi + yaw_alpha * dpsi)
    return (x + dx, y + dy, mpsi)

# ---------------- Mode mapping ----------------
def _apply_gt_mode():
    # Translate a human-readable GT_MODE into the internal boolean knobs used
    # across the code. Keeps centralized control of mode semantics.
    global STRICT_GT_YAW_LOCK, STRICT_GT_SPEED_LOCK, STRICT_GT_SOFT_POS
    global USE_GT_INIT, USE_GT_LEARN_VSCALE, USE_GT_GYRO_BIAS, USE_GT_ADAPT_RV

    if GT_MODE == "off":
        STRICT_GT_YAW_LOCK   = False
        STRICT_GT_SPEED_LOCK = False
        STRICT_GT_SOFT_POS   = False
        USE_GT_INIT          = False
        USE_GT_LEARN_VSCALE  = False
        USE_GT_GYRO_BIAS     = False
        USE_GT_ADAPT_RV      = True
    elif GT_MODE == "assist":
        STRICT_GT_YAW_LOCK   = False
        STRICT_GT_SPEED_LOCK = False
        STRICT_GT_SOFT_POS   = True
        USE_GT_INIT          = True
        USE_GT_LEARN_VSCALE  = True
        USE_GT_GYRO_BIAS     = True
        USE_GT_ADAPT_RV      = True
    elif GT_MODE == "strict":
        STRICT_GT_YAW_LOCK   = True
        STRICT_GT_SPEED_LOCK = True
        STRICT_GT_SOFT_POS   = True
        USE_GT_INIT          = True
        USE_GT_LEARN_VSCALE  = True
        USE_GT_GYRO_BIAS     = True
        USE_GT_ADAPT_RV      = True
    else:
        print(f"[StepE] Unknown GT_MODE='{GT_MODE}', defaulting to 'assist'")
        GT_MODE == "assist"
        _apply_gt_mode()

# ---------------- Fixed-rate controller resampler ----------------
class ControllerBus:
    """
    Buffer irregular incoming EKF states, interpolate in time, predict forward
    by configured latency, smooth outputs and publish a stable UDP JSON payload
    at a fixed rate. This class is deliberately self-contained and does not
    require tight synchronization with the rest of the system.
    """
    def __init__(self, host: str, port: int, hz: float):
        self.hz = float(hz)
        self.dt = 1.0 / max(1e-6, self.hz)
        self.host, self.port = host, port
        self.sock = None
        if CTRL_STREAM_ENABLE:
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                print(f"[StepE] CTRL bus → {host}:{port} @ {hz:.0f} Hz (smoothed + predicted)")
            except Exception as e:
                print(f"[StepE] CTRL bus disabled ({e})")
                self.sock = None

        # circular buffer sized based on desired buffer seconds / dt
        self.buf = deque(maxlen=int(max(3, CTRL_STATE_BUFFER_S / self.dt)))
        self.next_pub_t = None   # sim-time next publish timestamp
        # smoothing state
        self.v_sm = None
        self.psi_sm = None
        # last published for rate limits
        self.last_pub_simt = None
        self.last_pub_v = None
        self.last_pub_psi = None

    @staticmethod
    def _anglerp(a0, a1, w):
        # angle-aware linear interpolation (wrap-corrected)
        da = _wrap(a1 - a0)
        return _wrap(a0 + w * da)

    def push(self, t, x, y, psi, v):
        # Buffer an incoming sample: t should be sim-time consistent with others
        self.buf.append((float(t), float(x), float(y), float(psi), float(v)))

    def _interp_state(self, target_t):
        """
        Interpolate a buffered time-series to target_t (sim-time).
        Returns (x, y, psi, v, wz, a_long) where wz and a_long are finite-diff estimates.
        If no buffer exists, returns None.
        """
        if not self.buf:
            return None
        # find the latest sample at or before target_t (walk backwards for efficiency)
        before = None; after = None
        for i in range(len(self.buf)-1, -1, -1):
            ti = self.buf[i][0]
            if ti <= target_t:
                before = self.buf[i]
                after  = self.buf[i+1] if i+1 < len(self.buf) else self.buf[i]
                break
        if before is None:
            before = self.buf[0]; after = self.buf[0]
        t0,x0,y0,psi0,v0 = before
        t1,x1,y1,psi1,v1 = after
        if t1 <= t0 or target_t <= t0:
            w = 0.0
        elif target_t >= t1:
            w = 1.0
        else:
            w = (target_t - t0) / max(1e-9, (t1 - t0))
        xi = x0 + w*(x1 - x0)
        yi = y0 + w*(y1 - y0)
        psii = self._anglerp(psi0, psi1, w)
        vi = v0 + w*(v1 - v0)
        # derivatives for forward prediction (finite differences)
        dt = max(1e-3, (t1 - t0))
        w_gyro = _wrap(psi1 - psi0) / dt
        a_long = (v1 - v0) / dt
        return (xi, yi, psii, vi, w_gyro, a_long)

    # forward predict by latency using simple unicycle + linear accel
    @staticmethod
    def _predict(x, y, psi, v, w, a, dt):
        # Integrate kinematics using midpoint (Heun) to limit integration error.
        psi_f = psi + w * dt
        v_f   = v + a * dt
        v_avg = 0.5 * (v + v_f)
        psi_mid = psi + 0.5 * w * dt
        x_f = x + v_avg * math.cos(psi_mid) * dt
        y_f = y + v_avg * math.sin(psi_mid) * dt
        return x_f, y_f, _wrap(psi_f), v_f

    def maybe_publish(self, sim_now, sim_latency, csv_writer, ti_for_csv):
        """
        Publish at fixed sim-time grid while sim_now >= next_pub_t.
        Steps:
          - interpolate to a target (slightly past) timestamp within buffer
          - predict forward by sim_latency to produce actuation-time state
          - apply EMA smoothing on speed and yaw
          - enforce rate limits on yaw and speed changes
          - send UDP JSON (non-blocking best-effort)
        """
        if self.sock is None: return
        if self.next_pub_t is None:
            # align first publication on current sim time
            self.next_pub_t = sim_now
        # publish at fixed sim-time grid (may catch up multiple slots)
        while sim_now + 1e-6 >= self.next_pub_t:
            target_t = self.next_pub_t - sim_latency  # get past sample then predict forward
            st = self._interp_state(target_t)
            if st is None:
                self.next_pub_t += self.dt
                continue
            x, y, psi, v, wz, a = st

            # forward prediction to actuation horizon
            x_p, y_p, psi_p, v_p = self._predict(x, y, psi, v, wz, a, sim_latency)

            # smoothing (EMA); tau controls cutoff frequency
            def ema(prev, new, tau):
                if prev is None: return new
                alpha = self.dt / (tau + self.dt)
                return prev + alpha * (new - prev)
            self.v_sm   = ema(self.v_sm, v_p, CTRL_TAU_V_S)
            # yaw EMA in angle space
            if self.psi_sm is None:
                self.psi_sm = psi_p
            else:
                d = _wrap(psi_p - self.psi_sm)
                alpha = self.dt / (CTRL_TAU_YAW_S + self.dt)
                self.psi_sm = _wrap(self.psi_sm + alpha * d)

            # rate limiting: limit change since last published state
            if self.last_pub_simt is not None:
                dts = max(1e-3, self.next_pub_t - self.last_pub_simt)
                # yaw limit (convert CTRL_MAX_DPS -> radians per dts)
                dpsi_max = math.radians(CTRL_MAX_DPS) * dts
                dpsi = _wrap(self.psi_sm - self.last_pub_psi)
                dpsi = _clamp(dpsi, -dpsi_max, dpsi_max)
                psi_pub = _wrap(self.last_pub_psi + dpsi)
                # v limit
                dv_max = CTRL_MAX_ACC * dts
                dv = self.v_sm - self.last_pub_v
                dv = _clamp(dv, -dv_max, dv_max)
                v_pub = self.last_pub_v + dv
            else:
                psi_pub = self.psi_sm
                v_pub   = self.v_sm

            # assemble payload (timestamped for sim timeline)
            payload = {"t": float(target_t + sim_latency), "x": float(x_p), "y": float(y_p),
                       "psi": float(psi_pub), "v": float(v_pub)}
            try:
                self.sock.sendto(json.dumps(payload).encode("utf-8"),
                                 (self.host, self.port))
            except Exception:
                # best-effort send; don't let network errors break the control loop
                pass

            # Optional CSV hook for debugging: non-blocking logging
            if csv_writer is not None:
                csv_writer.writerow([
                    ti_for_csv,  # EKF timestamp for reference
                    float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"),
                    "", float("nan"), 0,
                    float("nan"), float("nan"), float("nan"),
                    float("nan"), float("nan"), float("nan"),
                    float("nan"), 0, "0.0",
                    float("nan"), float("nan"), float("nan"), float("nan"), float("nan"),
                    float("nan"),
                    float(target_t + sim_latency), x_p, y_p, psi_pub, v_pub
                ])

            # update publish bookkeeping and step to next tick
            self.last_pub_simt = self.next_pub_t
            self.last_pub_psi  = psi_pub
            self.last_pub_v    = v_pub
            self.next_pub_t   += self.dt

# ---------------- Main ----------------
LOGGING = True

def main():
    # apply GT mode into internal knobs
    global LOGGING
    _apply_gt_mode()

    # -------- Sensors & preprocessing ----------
    # Readers abstract the simulator/sensor access (IMU, speed, lidar centroids)
    fsds_cfg = FSDSConfig(vehicle_name="FSCar", imu_name="Imu")
    readers = FSDSReaders(fsds_cfg)
    lidar   = LidarFrontend(vehicle_name="FSCar", lidar_name="Lidar")

    # -------- EKF initialisation ----------
    ekf = ESEKFSpine(SpineConfig())
    ekf.Rv = float(LP.Rv_baseline)      # wheel-speed variance baseline
    v_scale = float(LP.v_scale)         # learned wheel speed scale
    b_wz    = float(LP.gyro_bias)       # learned gyro bias

    # -------- Centerline (map) ----------
    cl = build_centerline(PATHPOINTS_CSV, transform=True, is_loop=False)

    # -------- Optional GT client (FSDS) ----------
    gt_cli = None
    if _FSDS_OK and GT_MODE in ("assist", "strict"):
        try:
            gt_cli = fsds.FSDSClient(); gt_cli.confirmConnection()
            print(f"[StepE] GT supervision enabled (mode='{GT_MODE}').")
        except Exception as e:
            print(f"[StepE] GT supervision disabled ({e}).")
            gt_cli = None

    # Prepare logging, plotting and optional streams
    _ensure_logs()
    plot = LivePlot(cl.xy)

    # Legacy UDP stream (unchanged behaviour for compatibility)
    sock = None
    if STREAM_ENABLE:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            print(f"[StepE] UDP stream → {STREAM_HOST}:{STREAM_PORT} @ {STREAM_HZ:.0f} Hz (predictions only)")
        except Exception as e:
            print(f"[StepE] Stream disabled ({e})")
            sock = None

    # Controller bus that publishes a fixed-rate, smoothed, latency-compensated state
    ctrl_bus = ControllerBus(CTRL_STREAM_HOST, CTRL_STREAM_PORT, CTRL_STREAM_HZ) if CTRL_STREAM_ENABLE else None

    # ------------- runtime state ----------------
    last_imu_t = None
    last_mono  = time.monotonic()
    last_reg   = 0.0
    last_idx   = None
    last_err_idx = None
    last_plot  = 0.0
    last_stream = 0.0

    gt_ready = False
    first_imu_stamp = None
    last_gt_yaw = None
    last_gt_t   = None
    last_nudge  = 0.0
    rv_adapt    = 0.0
    last_persist = time.monotonic()

    # s-progress gate state
    last_s_ok   = None
    last_reg_ok_time = None

    # controller time mapping (sim time ↔ monotonic time)
    sim_to_mono_offset = None  # mono - sim
    sim_latency = CTRL_LATENCY_S

    # Precompute tightly yaw-locked covariances used in strict mode updates
    R3_yawlock = np.diag([100.0, 100.0, np.deg2rad(GT_YAW_SIG_DEG)**2])   # huge x,y → yaw-only
    R3_softpos = np.diag([GT_POS_R_XY**2, GT_POS_R_XY**2, np.deg2rad(GT_POS_R_YAW_DEG)**2])

    print(f"[StepE] Running (mode='{GT_MODE}'). Ctrl+C to stop.")
    try:
        with open(LOG_FILE, "a", newline="") as fcsv:
            w = csv.writer(fcsv)

            # buffer for controller resampler (constructed earlier if enabled)
            if ctrl_bus is not None:
                pass

            # Main loop — driven by matplotlib window existence for live plotting.
            while plt.fignum_exists(plot.fig.number):
                # ---------- IMU predict ----------
                wz_raw, ax, ti = readers.imu()
                now_mono = time.monotonic()
                if first_imu_stamp is None: first_imu_stamp = ti

                # compute dt_used from timestamps; handle duplicates or missing stamps
                dup_flag = 0
                if last_imu_t is None:
                    dt_used = 0.0
                else:
                    dt_stamp = ti - last_imu_t
                    dt_mono  = now_mono - last_mono
                    if dt_stamp is None or dt_stamp <= 0.0 or dt_stamp < 1e-6:
                        # stamp invalid → use wall-clock dt, mark duplicate
                        dup_flag = 1
                        dt_used = max(1e-4, dt_mono)
                    else:
                        dt_used = dt_stamp
                last_imu_t, last_mono = ti, now_mono

                # maintain a smoothed mapping between sim-time (ti) and monotonic wall time
                if sim_to_mono_offset is None and ti is not None:
                    sim_to_mono_offset = now_mono - ti
                elif ti is not None:
                    sim_to_mono_offset = 0.98*sim_to_mono_offset + 0.02*(now_mono - ti)

                # Predict step of EKF using gyro (with optional bias subtraction)
                if USE_GT_GYRO_BIAS and (last_gt_yaw is not None) and (last_gt_t is not None) and dt_used > 1e-3:
                    # placeholder for possible bias logic that depends on GT yaw rate; updated later
                    yawdot_gt = _wrap(last_gt_yaw - last_gt_yaw)
                # subtract learned/applying gyro bias if enabled
                wz = (wz_raw - b_wz) if USE_GT_GYRO_BIAS else wz_raw
                ekf.predict(dt=dt_used, wz_imu=wz, ax_imu=ax)

                # ---------- Speed update (wheel) ----------
                # Apply learned v_scale before fusing wheel speed into the EKF
                v_meas, tv = readers.speed()
                v_meas_raw = float(v_meas)
                v_meas *= float(v_scale)      # correct sensor scale bias
                age = abs(ti - tv)
                _, nis_v = ekf.update_speed(v_meas=v_meas, age_sec=age)

                # ---------- Ground-truth usage (optional) ----------
                gt_x = gt_y = gt_psi = float("nan"); v_gt = float("nan")
                if gt_cli is not None:
                    st = gt_cli.getCarState()
                    pos = st.kinematics_estimated.position
                    ori = st.kinematics_estimated.orientation
                    lin = st.kinematics_estimated.linear_velocity
                    gt_x = float(pos.x_val); gt_y = float(pos.y_val); gt_psi = _yaw_from_quat(ori)
                    v_gt = float(math.hypot(lin.x_val, lin.y_val))

                    # On first GT sighting, optionally flip centerline direction to match spawn heading
                    if USE_GT_INIT and not gt_ready:
                        _maybe_flip_centerline_direction_by_gt(cl, gt_x, gt_y, gt_psi)
                        gt_ready = True

                    # Strict yaw locking: replace EKF yaw with GT yaw but keep EKF x,y
                    if GT_MODE == "strict":
                        x_est, y_est, psi_est, *_ = ekf.state()
                        ekf.update_pose2d(x_est, y_est, gt_psi, R3_yawlock)

                    # Fast online gyro bias learning using GT yaw rate (if enabled)
                    if USE_GT_GYRO_BIAS and (last_gt_yaw is not None) and (last_gt_t is not None) and dt_used > 1e-3:
                        yawdot_gt = _wrap(gt_psi - last_gt_yaw) / max(1e-3, (ti - last_gt_t))
                        beta = GYRO_BIAS_BETA_FAST if (ti - first_imu_stamp) <= GYRO_BIAS_FAST_S else GYRO_BIAS_BETA_SLOW
                        err = (wz_raw - yawdot_gt)
                        b_wz = (1.0 - beta)*b_wz + beta*err

                    # Online v_scale adaptation if GT speed available
                    if USE_GT_LEARN_VSCALE and v_gt > 0.5 and v_meas_raw > 0.1:
                        r = np.clip(v_gt / max(1e-3, v_meas_raw), VSCALE_CLAMP[0], VSCALE_CLAMP[1])
                        v_scale = float((1.0 - VSCALE_ALPHA)*v_scale + VSCALE_ALPHA*r)

                    # Strict mode: temporarily lower speed noise to fuse GT speed tightly
                    if GT_MODE == "strict" and np.isfinite(v_gt):
                        Rv_old = ekf.Rv
                        ekf.Rv = 0.03
                        ekf.update_speed(v_meas=v_gt, age_sec=0.0)
                        ekf.Rv = Rv_old

                    last_gt_yaw = gt_psi; last_gt_t = ti

                # ---------- Rv adaptation from NIS ----------
                # Keep wheel-speed variance adapted so NIS_v hovers near target.
                if USE_GT_ADAPT_RV and hasattr(ekf, "_nis_win") and len(ekf._nis_win) >= 6:
                    avg = float(np.mean(ekf._nis_win))
                    if np.isfinite(avg) and avg > 0.05:
                        scale = (avg / NISV_TARGET)
                        ekf.Rv = float(np.clip(ekf.Rv * (1.0 + NISV_ALPHA*(scale - 1.0)),
                                               RV_CLAMP[0], RV_CLAMP[1]))
                        rv_adapt = ekf.Rv

                # ---------- LiDAR→centerline registration (periodic) ----------
                reg_mode = "none"; reg_nis = float("nan"); reg_ms = 0.0
                C_body = None
                if (time.monotonic() - last_reg) >= REG_DT:
                    last_reg = time.monotonic()
                    C_body = lidar.get_centroids_body()
                    if C_body.shape[0] >= CENT_MIN:
                        # subsample if too many centroids to keep runtime bounded
                        if C_body.shape[0] > CENT_MAX:
                            stride = max(1, C_body.shape[0] // CENT_MAX)
                            C_body = C_body[::stride]

                        x, y, psi, *_ = ekf.state()
                        t0 = time.perf_counter()
                        # robust registration function returns corrected pose + measurement cov R3 + idx
                        x_corr, y_corr, psi_corr, R3, new_idx = robust_pose_correction_centerline_win(
                            C_body, cl, last_idx, WINDOW_M,
                            x, y, psi,
                            max_iters=6, trim_frac=0.35,
                            huber_c_lat=0.22, huber_c_psi=np.deg2rad(4.0)
                        )
                        reg_ms = (time.perf_counter() - t0) * 1000.0

                        # curvature-aware measurement covariance adjustments:
                        # increase pos variance and reduce yaw var on sharp curves (be conservative)
                        curv = _curvature_at_idx(cl, new_idx if new_idx is not None else (last_idx or 0))
                        R3_use = R3.copy()
                        if curv >= CURV_STRONG:
                            R3_use[0,0] *= 2.2; R3_use[1,1] *= 2.2; R3_use[2,2] *= 0.7
                            yaw_cap = np.deg2rad(5.5); pos_cap = 0.45
                        else:
                            R3_use[0,0] *= 0.9; R3_use[1,1] *= 0.9; R3_use[2,2] *= 1.2
                            yaw_cap = INNOV_CLAMP_YAW; pos_cap = INNOV_CLAMP_POS

                        # NIS gating: compute full pose NIS and also a yaw-only NIS
                        yv_full = np.array([x_corr - x, y_corr - y, _wrap(psi_corr - psi)], dtype=float)
                        nis_full = _predicted_nis_pose3(ekf.covariance(), yv_full, R3_use)
                        R3_yaw = R3_use.copy(); R3_yaw[0,0] *= 25.0; R3_yaw[1,1] *= 25.0
                        yv_yaw = np.array([0.0, 0.0, _wrap(psi_corr - psi)], dtype=float)
                        nis_yaw = _predicted_nis_pose3(ekf.covariance(), yv_yaw, R3_yaw)

                        # s-progress gate: prevent weird jumps in s along centerline
                        s_corr, _ = _s_at_xy(cl, x_corr, y_corr, idx_hint=new_idx, window_m=6.0)
                        s_gate_ok = True
                        if last_s_ok is not None and last_reg_ok_time is not None:
                            dt_reg = max(1e-3, (time.monotonic() - last_reg_ok_time))
                            v_now  = max(0.0, float(ekf.state()[3]))
                            ds_max = v_now * dt_reg * S_GATE_RATE_SCALE + S_GATE_MARGIN_M
                            ds = abs(s_corr - last_s_ok)
                            if ds > ds_max: s_gate_ok = False

                        # Apply registration update depending on NIS gating outcome
                        if s_gate_ok and (nis_full < FULL2YAW_NIS):
                            # full pose update (clamped)
                            x_est, y_est, psi_est, *_ = ekf.state()
                            if GT_MODE == "strict":
                                # in strict GT mode, we keep EKF psi unchanged here and only update XY
                                mx, my, _ = _limit_pose_innovation(
                                    x_est, y_est, psi_est, x_corr, y_corr, psi_est,
                                    yaw_cap=yaw_cap, pos_cap=pos_cap, yaw_alpha=YAW_EMA_ALPHA_REG
                                )
                                ekf.update_pose2d(mx, my, psi_est, R3_use)
                            else:
                                mx, my, mpsi = _limit_pose_innovation(
                                    x_est, y_est, psi_est, x_corr, y_corr, psi_corr,
                                    yaw_cap=yaw_cap, pos_cap=pos_cap, yaw_alpha=YAW_EMA_ALPHA_REG
                                )
                                ekf.update_pose2d(mx, my, mpsi, R3_use)
                            reg_mode = "full"; reg_nis = nis_full; last_idx = new_idx
                            last_s_ok = s_corr; last_reg_ok_time = time.monotonic()

                        elif s_gate_ok and (nis_yaw < YAW2SKIP_NIS):
                            # yaw-only update allowed even if full pose is too inconsistent
                            reg_mode = "yaw"; reg_nis = nis_yaw; last_idx = new_idx
                            last_reg_ok_time = time.monotonic()
                        else:
                            # skip update: registration not trusted
                            reg_mode = "skip"; reg_nis = max(nis_full, nis_yaw)

                # ---------- Strong GT yaw leash (runs every IMU tick if GT present) ----------
                # Heavily favors correcting heading using GT yaw while keeping X/Y variance huge.
                yaw_err_deg_log = float("nan")
                if gt_cli is not None and np.isfinite(gt_psi):
                    x_est, y_est, psi_est, v_est, *_ = ekf.state()
                    curv_here = _curvature_at_idx(cl, last_err_idx if last_err_idx is not None else 0)

                    # choose correction regime: strict early phase or curvature-based gains
                    strict_phase = (first_imu_stamp is not None) and ((ti - first_imu_stamp) <= YAW_STRICT_WINDOW_S)
                    deadband = YAW_DEADBAND_DEG
                    if strict_phase:
                        sigma = YAW_SIG_STRICT_DEG
                        ema_a = YAW_EMA_STRICT
                        step_cap = _rad(YAW_STEP_STRICT_DEG)
                    else:
                        if curv_here <= YAW_CURV_WEAK:
                            sigma = YAW_SIG_STRAIGHT_DEG
                            ema_a = YAW_EMA_ALPHA_STRONG
                            step_cap = _rad(YAW_STEP_CAP_DEG)
                        elif curv_here >= YAW_CURV_STRONG:
                            sigma = YAW_SIG_CORNER_DEG
                            ema_a = 0.45
                            step_cap = _rad(0.75 * YAW_STEP_CAP_DEG)
                        else:
                            # smooth interpolation between straight/corner regimes
                            r = 1.0 - _interp01(curv_here, YAW_CURV_WEAK, YAW_CURV_STRONG)
                            sigma = YAW_SIG_CORNER_DEG * (1.0 - r) + YAW_SIG_STRAIGHT_DEG * r
                            ema_a = 0.45 + (YAW_EMA_ALPHA_STRONG - 0.45) * r
                            step_cap = _rad(0.75*YAW_STEP_CAP_DEG + 0.25*YAW_STEP_CAP_DEG*r)

                    # apply correction if outside deadband
                    epsi = _wrap(gt_psi - psi_est)
                    if abs(_deg(epsi)) > deadband:
                        epsi_c = _clamp(epsi, -step_cap, step_cap)
                        dpsi = ema_a * epsi_c
                        psi_target = _wrap(psi_est + dpsi)
                        R_yawonly = _yaw_only_R3(sigma)
                        ekf.update_pose2d(x_est, y_est, psi_target, R_yawonly)

                        # micro-iterations: apply a couple of small extra steps if still large on straights
                        for _ in range(YAW_MICRO_ITERS_MAX):
                            resid = _wrap(gt_psi - ekf.state()[2])
                            if abs(_deg(resid)) < 2.0 or curv_here > YAW_CURV_WEAK:
                                break
                            step = ema_a * _clamp(resid, -0.5*step_cap, 0.5*step_cap)
                            ekf.update_pose2d(x_est, y_est, _wrap(ekf.state()[2] + step), R_yawonly)

                    yaw_err_deg_log = _deg(_wrap(gt_psi - ekf.state()[2]))

                # ---------- Soft position leash to GT (assist/strict) ----------
                x, y, psi, v, *_ = ekf.state()
                if STRICT_GT_SOFT_POS and (gt_cli is not None) and np.isfinite(gt_x) and np.isfinite(gt_y):
                    dx = x - gt_x; dy = y - gt_y
                    if math.hypot(dx, dy) > GT_POS_SOFT_M and (time.monotonic() - last_nudge) >= GT_NUDGE_DT:
                        # softly nudge pose to GT to prevent long-running XY drift
                        ekf.update_pose2d(gt_x, gt_y, psi if GT_MODE=="strict" else psi, R3_softpos)
                        last_nudge = time.monotonic()
                        x, y, psi, v, *_ = ekf.state()

                # ---------- Diagnostics: lateral/heading errors ----------
                e_lat, e_yaw, s_here, last_err_idx = _signed_lateral_error(
                    cl, x, y, psi, idx_guess=last_err_idx, window_m=8.0
                )

                # Persist learned baselines periodically to stabilize future runs
                if PERSIST_LEARNING and (time.monotonic() - last_persist) >= PERSIST_EVERY_S:
                    last_persist = time.monotonic()
                    LP.smooth_update(Rv=ekf.Rv, gyro_bias=b_wz, v_scale=v_scale, alpha=0.35)
                    LP.save(LP_PATH)

                # Log (append) a compact CSV row for offline inspection
                if LOGGING:
                    w.writerow([
                        ti, x, y, psi, v,
                        (np.mean(ekf._nis_win) if getattr(ekf, "_nis_win", None) else float("nan")), ekf.Rv,
                        reg_mode, reg_nis, int(C_body.shape[0]) if C_body is not None else 0,
                        e_lat, e_yaw, s_here,
                        gt_x, gt_y, gt_psi,
                        dt_used, dup_flag, f"{reg_ms:.2f}",
                        rv_adapt, b_wz, v_scale, LP.Rv_baseline, LP.gyro_bias,
                        yaw_err_deg_log, float("nan"), float("nan"), float("nan"), float("nan")
                    ])

                # ---------- Controller bus: push state + publish at fixed rate ----------
                if ctrl_bus is not None and sim_to_mono_offset is not None:
                    # push latest EKF state into controller buffer (sim time axis)
                    ctrl_bus.push(ti, x, y, psi, v)
                    # compute "sim now" from monotonic time and request publishing
                    sim_now = time.monotonic() - sim_to_mono_offset
                    ctrl_bus.maybe_publish(sim_now, sim_latency, None, ti_for_csv=ti)

                # ---------- Plot update ----------
                t_plot = time.monotonic()
                if (t_plot - last_plot) >= (1.0 / PLOT_HZ):
                    last_plot = t_plot
                    plot.update(
                        x, y, psi, e_lat, e_yaw, v,
                        (np.mean(ekf._nis_win) if getattr(ekf, "_nis_win", None) else float("nan")),
                        reg_mode, reg_nis
                    )

                # ---------- Legacy UDP predictions stream ----------
                t_stream = time.monotonic()
                if sock is not None and (t_stream - last_stream) >= (1.0 / STREAM_HZ):
                    last_stream = t_stream
                    try:
                        payload = {"t": float(ti), "x": float(x), "y": float(y),
                                   "psi": float(psi), "v": float(v)}
                        sock.sendto(json.dumps(payload).encode("utf-8"),
                                    (STREAM_HOST, STREAM_PORT))
                    except Exception:
                        pass

                # small sleep to avoid busy-looping; real-time behaviour depends on sensor rate
                time.sleep(0.001)

    except KeyboardInterrupt:
        print("\n[StepE] Stopping…")
    finally:
        # Save learned params on exit (conservative smoothing)
        try:
            if PERSIST_LEARNING:
                LP.smooth_update(Rv=ekf.Rv, gyro_bias=b_wz, v_scale=v_scale, alpha=0.5)
                LP.save(LP_PATH)
        except Exception:
            pass
        readers.release()
        print("[StepE] Logs at", LOG_FILE)

# ---------------- Plot (simple live UI) ----------------
class LivePlot:
    """
    Lightweight matplotlib-based live plot showing:
       - centerline
       - trajectory path
       - current heading vector and position
       - text diagnostics (NIS, reg mode, e_lat/e_yaw)
    Interaction keys:
       q = quit, c = toggle car-centered view, p = pause, l = toggle CSV logging,
       + / - change view scale, r = reset trajectory history.
    """
    def __init__(self, cl_xy: np.ndarray):
        matplotlib.use("TkAgg")
        self.fig, self.ax = plt.subplots(figsize=(8,8))
        self.ax.set_aspect("equal", adjustable="datalim")
        self.ax.grid(True, linestyle="--", alpha=0.3)
        self.ax.set_title("Step E — Trajectory vs Centerline")
        (self.cl_line,) = self.ax.plot(cl_xy[:,0], cl_xy[:,1], color="#5566ff", lw=1.5, alpha=0.6, label="centerline")
        self.path_x = deque(maxlen=TRAJ_BUFFER_N); self.path_y = deque(maxlen=TRAJ_BUFFER_N)
        (self.path_line,) = self.ax.plot([], [], color="#ff8c00", lw=2.0, alpha=0.9, label="trajectory")
        self.car_heading = self.ax.plot([], [], color="#00bcd4", lw=2.0)[0]
        self.car_dot = self.ax.scatter([], [], s=30, c="#00bcd4", zorder=5)
        self.txt = self.ax.text(0.02, 0.98, "", transform=self.ax.transAxes, va="top", ha="left",
                                fontsize=10, bbox=dict(facecolor="white", alpha=0.75, boxstyle="round,pad=0.3"))
        self.ax.legend(loc="lower right")
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.use_car_center = USE_CAR_CENTER
        self.view_half = CAR_VIEW_HALF
        self.paused = PAUSE_PLOT

    def on_key(self, ev):
        global LOGGING
        if ev.key == "q": plt.close(self.fig)
        elif ev.key == "c": self.use_car_center = not self.use_car_center
        elif ev.key == "p": self.paused = not self.paused
        elif ev.key == "l": LOGGING = not LOGGING; print(f"[StepE] CSV logging {'ON' if LOGGING else 'OFF'}")
        elif ev.key == "+": self.view_half = min(150.0, self.view_half + 5.0)
        elif ev.key == "-": self.view_half = max(8.0, self.view_half - 5.0)
        elif ev.key == "r": self.path_x.clear(); self.path_y.clear()

    def update(self, x, y, psi, e_lat, e_yaw, v, nis_v, reg_mode, reg_nis):
        if self.paused:
            plt.pause(0.001); return
        self.path_x.append(x); self.path_y.append(y)
        self.path_line.set_data(self.path_x, self.path_y)
        hx = [x, x + CAR_HEADING_M * math.cos(psi)]
        hy = [y, y + CAR_HEADING_M * math.sin(psi)]
        self.car_heading.set_data(hx, hy)
        self.car_dot.set_offsets(np.array([[x, y]], dtype=float))
        self.txt.set_text(
            f"x={x:6.2f}  y={y:6.2f}  ψ={math.degrees(psi):6.1f}°  v={v:4.1f} m/s\n"
            f"e_lat={e_lat:+5.2f} m   e_yaw={math.degrees(e_yaw):+5.1f}°\n"
            f"NIS_v≈{nis_v:4.1f}   regNIS={reg_nis:5.1f} ({reg_mode})   CSV={'ON' if LOGGING else 'OFF'}"
        )
        if self.use_car_center:
            r = self.view_half
            self.ax.set_xlim(x - r, x + r); self.ax.set_ylim(y - r, y + r)
        else:
            self.ax.relim(); self.ax.autoscale_view()
        self.fig.canvas.draw_idle(); plt.pause(0.001)

if __name__ == "__main__":
    main()
