import time
import math
import numpy as np
import pandas as pd
import fsds
from scipy.interpolate import splprep, splev

# -------------------- Setup --------------------
client = fsds.FSDSClient()
client.reset()
client.confirmConnection()
client.enableApiControl(True)

PATHPOINTS_CSV = "./data/vd_pathpoints_looped.csv"
ROUTE_IS_LOOP = False 
scaling_factor = 1

SEARCH_BACK = 10
SEARCH_FWD = 250
MAX_STEP = 60

WHEELBASE_M = 1.5
MAX_STEER_RAD = 0.6
LD_BASE = 2.5
LD_GAIN = 0.5
LD_MIN = 2.0
LD_MAX = 15.0

V_MAX = 50.0
AY_MAX = 5.0
AX_MAX = 5.0
AX_MIN = -4.0

PROFILE_WINDOW_M = 120.0
BRAKE_EXTEND_M = 60.0
NUM_ARC_POINTS = 1200
PROFILE_HZ = 5.0
BRAKE_GAIN = 0.6

STOP_SPEED_THRESHOLD = 0.1   # m/s, vehicle considered stopped

# -------------------- Utility Functions --------------------
def preprocess_path(xs, ys, loop=True):
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    x_next = np.roll(xs, -1) if loop else np.concatenate((xs[1:], xs[-1:]))
    y_next = np.roll(ys, -1) if loop else np.concatenate((ys[1:], ys[-1:]))
    seglen = np.hypot(x_next - xs, y_next - ys)
    if not loop:
        seglen[-1] = 0.0
    s = np.concatenate(([0.0], np.cumsum(seglen[:-1])))
    return xs, ys, s, float(seglen.sum())

def get_xy_speed(state):
    pos = state.kinematics_estimated.position
    return (pos.x_val, pos.y_val), float(getattr(state, "speed", 0.0))

def local_closest_index(xy, xs, ys, cur_idx, loop=True):
    x0, y0 = xy
    N = len(xs)
    if N == 0:
        return 0
    if loop:
        start = (cur_idx - SEARCH_BACK) % N
        count = min(N, SEARCH_BACK + SEARCH_FWD + 1)
        idxs = (np.arange(start, start + count) % N)
        dx, dy = xs[idxs] - x0, ys[idxs] - y0
        j = int(np.argmin(dx * dx + dy * dy))
        return int(idxs[j])
    else:
        i0 = max(0, cur_idx - SEARCH_BACK)
        i1 = min(N, cur_idx + SEARCH_FWD + 1)
        dx, dy = xs[i0:i1] - x0, ys[i0:i1] - y0
        j = int(np.argmin(dx * dx + dy * dy))
        return i0 + j

def calc_lookahead(speed_mps):
    return max(LD_MIN, min(LD_MAX, LD_BASE + LD_GAIN * speed_mps))

def get_yaw(state):
    q = state.kinematics_estimated.orientation
    w, x, y, z = q.w_val, q.x_val, q.y_val, q.z_val
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

def forward_index_by_distance(near_idx, Ld, s, total_len, loop=True):
    if loop:
        target = (s[near_idx] + Ld) % total_len
        return int(np.searchsorted(s, target, side="left") % len(s))
    else:
        target = min(s[near_idx] + Ld, s[-1])
        return int(np.searchsorted(s, target, side="left"))

def pure_pursuit_steer(pos_xy, yaw, speed, xs, ys, near_idx, s, total_len, loop=True):
    Ld = calc_lookahead(speed)
    tgt_idx = forward_index_by_distance(near_idx, Ld, s, total_len, loop)
    tx, ty = xs[tgt_idx], ys[tgt_idx]
    dx, dy = tx - pos_xy[0], ty - pos_xy[1]
    cy, sy = math.cos(yaw), math.sin(yaw)
    x_rel, y_rel = cy * dx + sy * dy, -sy * dx + cy * dy
    kappa = 2.0 * y_rel / max(0.5, Ld) ** 2
    delta = math.atan(WHEELBASE_M * kappa)
    return max(-1, min(1, delta / MAX_STEER_RAD)), tgt_idx

def resample_track(x_raw, y_raw, num_arc_points=NUM_ARC_POINTS):
    tck, _ = splprep([x_raw, y_raw], s=0, k=min(3, max(1, len(x_raw) - 1)))
    tt_dense = np.linspace(0, 1, 2000)
    xx, yy = splev(tt_dense, tck)
    s_dense = np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(xx), np.diff(yy)))))
    s_dense /= s_dense[-1] if s_dense[-1] > 0 else 1.0
    s_uniform = np.linspace(0, 1, num_arc_points)
    return np.interp(s_uniform, s_dense, xx), np.interp(s_uniform, s_dense, yy)

def compute_curvature(x, y):
    dx, dy = np.gradient(x), np.gradient(y)
    ddx, ddy = np.gradient(dx), np.gradient(dy)
    denom = np.power(dx*dx + dy*dy, 1.5)
    curv = np.abs(dx * ddy - dy * ddx) / (denom + 1e-12)
    curv[~np.isfinite(curv)] = 0.0
    return curv

def curvature_speed_limit(curvature):
    return np.minimum(np.sqrt(AY_MAX / (curvature + 1e-9)), V_MAX)

def profile_window(v_limit_win, ds_win, v0):
    Nw = len(v_limit_win)
    vp = np.zeros(Nw)
    vp[0] = min(v_limit_win[0], v0)

    # forward pass
    for i in range(1, Nw):
        vp[i] = min(math.sqrt(vp[i-1]**2 + 2 * AX_MAX * ds_win[i-1]), v_limit_win[i])

    # force stop at the end (only if non-loop route)
    vp[-1] = 0.0

    # backward pass (braking feasibility)
    for i in range(Nw - 2, -1, -1):
        vp[i] = min(vp[i], math.sqrt(vp[i+1]**2 + 2 * abs(AX_MIN) * ds_win[i]), v_limit_win[i])

    return vp

class PID:
    def __init__(self, kp, ki, kd):
        self.kp, self.ki, self.kd = kp, ki, kd
        self._i = 0.0
        self._prev_err = None
    def reset(self):
        self._i, self._prev_err = 0.0, None
    def update(self, err, dt):
        dt = max(dt, 1e-3)
        self._i += err * dt
        d = 0 if self._prev_err is None else (err - self._prev_err) / dt
        self._prev_err = err
        return max(0, min(1, self.kp * err + self.ki * self._i + self.kd * d))

# -------------------- Load path --------------------
df = pd.read_csv(PATHPOINTS_CSV)
rx, ry = resample_track(df["x"].to_numpy() * scaling_factor, 
                        df["y"].to_numpy() * scaling_factor)
route_x, route_y = ry + 15.0, -rx
route_x, route_y, route_s, route_len = preprocess_path(route_x, route_y, loop=ROUTE_IS_LOOP)
seg_ds = np.hypot(np.roll(route_x, -1) - route_x, np.roll(route_y, -1) - route_y)
v_limit_global = curvature_speed_limit(compute_curvature(route_x, route_y))
route_v = v_limit_global.copy()

# -------------------- Control Loop --------------------
th_pid = PID(3.2, 0.5, 2)
state = client.getCarState()
(cur_x, cur_y), speed = get_xy_speed(state)
cur_idx = int(np.argmin((route_x - cur_x)**2 + (route_y - cur_y)**2))

last_t = time.perf_counter()
last_profile_t = last_t

log = []

while True:
    state = client.getCarState()
    (cx, cy), speed = get_xy_speed(state)
    yaw = get_yaw(state)

    now = time.perf_counter()
    dt = now - last_t
    last_t = now

    cur_idx = local_closest_index((cx, cy), route_x, route_y, cur_idx)

    if now - last_profile_t >= (1.0 / PROFILE_HZ):
        last_profile_t = now
        v_prof = profile_window(v_limit_global, seg_ds, speed)
        route_v = v_prof

    steering, tgt_idx = pure_pursuit_steer((cx, cy), yaw, speed, route_x, route_y, cur_idx, route_s, route_len)

    v_err = route_v[cur_idx] - speed
    if v_err >= 0:
        throttle = th_pid.update(v_err, dt)
        brake = 0
    else:
        th_pid.reset()
        throttle, brake = 0, min(1, -v_err * BRAKE_GAIN)

    car_controls = fsds.CarControls()
    car_controls.throttle, car_controls.brake, car_controls.steering = throttle, brake, -steering * 1.2
    client.setCarControls(car_controls)

    # log data
    log.append([now, cx, cy, speed, throttle, brake, steering, v_err])
    if len(log) % 100 == 0:
        pd.DataFrame(log, columns=["t","x","y","speed","throttle","brake","steer","v_err"])\
          .to_csv("telemetry_log.csv", index=False)

    # --- Exit condition: vehicle has stopped at end of route ---
    if not ROUTE_IS_LOOP and cur_idx >= len(route_x) - 5 and speed < STOP_SPEED_THRESHOLD:
        print("Reached end of route and stopped. Exiting loop.")
        car_controls.handbrake = True
        client.setCarControls(car_controls)
        break

    time.sleep(0.02)
