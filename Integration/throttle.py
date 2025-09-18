# controls_from_localise.py
# Reads pose & speed from localise.py UDP stream and actuates via FSDS.
# Adds: constant-speed cruise mode, throttle/steer smoothing, stale-stream hold.

import time
import math
import threading
import tkinter as tk
from tkinter import ttk

import json
import socket

import numpy as np
import pandas as pd
import fsds
from scipy.interpolate import splprep, splev

# =========================================================
#                 LOCALISATION STREAM CONFIG
# =========================================================
LOC_HOST = "127.0.0.1"
LOC_PORT = 5602     # << you said you're on 5602
WARMUP_MIN_SAMPLES = 15
WARMUP_TIMEOUT_S   = None     # wait indefinitely
STALE_MAX_AGE_S    = 0.50     # a bit tolerant on Windows timers
SAFE_BRAKE         = 0.35

# =========================================================
#                 CRUISE / SMOOTHING CONFIG
# =========================================================
CRUISE_MODE        = True     # constant-speed mode (ignores curvature profile)
CRUISE_SPEED       = 0.1     # m/s — tune this (0.5–0.8 good for testing)
SPEED_DEADBAND     = 0.03     # m/s — avoid brake/throttle chatter
V_CMD_RAMP_A       = 0.30     # m/s^2 — target speed ramp limit

# PI gains (no D for smoothness)
PI_KP              = 1.6
PI_KI              = 0.4
PI_I_MAX           = 1.2      # integrator clamp
THROTTLE_SLEW_UP   = 1.2      # /s (units of [0..1] per sec)
THROTTLE_SLEW_DN   = 2.0      # /s (faster to back off)
BRAKE_GAIN         = 0.2      # brake from negative speed error

# Steering smoothing / rate limit (normalized -1..1)
STEER_GAIN         = 1.0      # was 1.2; keep calmer
STEER_SLEW         = 1.5      # /s (max change per second)
STEER_TAU          = 0.10     # s (EMA filter on steering cmd)

SPEED_TAU          = 0.20     # s (EMA on measured speed)

STOP_SPEED_THRESHOLD = 0.1

# =========================================================
#               LOCALISATION UDP SUBSCRIBER
# =========================================================
class LocaliseClient:
    """ Subscribes to UDP JSON datagrams: {"t","x","y","psi","v"} """
    def __init__(self, host=LOC_HOST, port=LOC_PORT):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        except Exception:
            pass
        # if you listen from any NIC: self.sock.bind(("0.0.0.0", port))
        self.sock.bind((host, port))
        self.sock.settimeout(0.2)

        self._latest = {"t": 0.0, "x": 0.0, "y": 0.0, "psi": 0.0, "v": 0.0}
        self._lock = threading.Lock()
        self._running = False

        self._last_rx_mono = None
        self._pkt_count = 0
        self._new_data_event = threading.Event()

    def start(self):
        if self._running: return
        self._running = True
        threading.Thread(target=self._rx_loop, daemon=True).start()

    def stop(self):
        self._running = False
        try: self.sock.close()
        except: pass

    def _rx_loop(self):
        while self._running:
            try:
                data, _ = self.sock.recvfrom(4096)
                msg = json.loads(data.decode("utf-8"))
                now = time.monotonic()
                with self._lock:
                    self._latest = msg
                    self._last_rx_mono = now
                    self._pkt_count += 1
                self._new_data_event.set()
            except socket.timeout:
                continue
            except Exception:
                time.sleep(0.02)

    def wait_for_stream(self, min_samples=WARMUP_MIN_SAMPLES, timeout=WARMUP_TIMEOUT_S, max_age_s=STALE_MAX_AGE_S):
        t0 = time.monotonic()
        while True:
            if (timeout is not None) and ((time.monotonic() - t0) >= timeout):
                return False
            self._new_data_event.wait(0.2)
            self._new_data_event.clear()
            with self._lock:
                count = self._pkt_count
                last_rx = self._last_rx_mono
            if count >= min_samples and last_rx is not None and (time.monotonic() - last_rx) <= max_age_s:
                return True

    def is_fresh(self, max_age_s=STALE_MAX_AGE_S):
        with self._lock:
            last_rx = self._last_rx_mono
        return (last_rx is not None) and ((time.monotonic() - last_rx) <= max_age_s)

    def get_state(self):
        with self._lock:
            m = dict(self._latest)
        return m.get("t", 0.0), m.get("x", 0.0), m.get("y", 0.0), m.get("psi", 0.0), m.get("v", 0.0)

# =========================================================
#                  CONTROLLER CONSTANTS
# =========================================================
PATHPOINTS_CSV = "./data/vd_pathpoints.csv"
ROUTE_IS_LOOP = False
scaling_factor = 1

SEARCH_BACK = 10
SEARCH_FWD = 250

WHEELBASE_M = 1.5
MAX_STEER_RAD = 0.6
LD_BASE = 2.5
LD_GAIN = 0.5
LD_MIN = 2.0
LD_MAX = 15.0

# (these curvature-based limits are not used in cruise mode)

# =========================================================
#                  UTILITIES (UNCHANGED)
# =========================================================
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
        j = int(np.argmin(dx*dx + dy*dy))
        return int(idxs[j])
    else:
        i0 = max(0, cur_idx - SEARCH_BACK)
        i1 = min(N, cur_idx + SEARCH_FWD + 1)
        dx, dy = xs[i0:i1] - x0, ys[i0:i1] - y0
        j = int(np.argmin(dx*dx + dy*dy))
        return i0 + j

def calc_lookahead(speed_mps):
    return max(LD_MIN, min(LD_MAX, LD_BASE + LD_GAIN * speed_mps))

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

def resample_track(x_raw, y_raw, num_arc_points=1600):
    tck, _ = splprep([x_raw, y_raw], s=0, k=min(3, max(1, len(x_raw) - 1)))
    tt_dense = np.linspace(0, 1, 2000)
    xx, yy = splev(tt_dense, tck)
    s_dense = np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(xx), np.diff(yy)))))
    s_dense /= s_dense[-1] if s_dense[-1] > 0 else 1.0
    s_uniform = np.linspace(0, 1, num_arc_points)
    return np.interp(s_uniform, s_dense, xx), np.interp(s_uniform, s_dense, yy)

# =========================================================
#              PATH LOADING (unchanged logic)
# =========================================================
def build_route():
    df = pd.read_csv(PATHPOINTS_CSV)
    rx, ry = resample_track(df["x"].to_numpy() * scaling_factor,
                            df["y"].to_numpy() * scaling_factor)
    route_x, route_y = ry + 15.0, -rx  # original transform
    route_x, route_y, route_s, route_len = preprocess_path(route_x, route_y, loop=ROUTE_IS_LOOP)
    return route_x, route_y, route_s, route_len

# =========================================================
#          WRAPPED CONTROL LOOP (reads from Localise)
# =========================================================
class PI:
    def __init__(self, kp, ki, i_max):
        self.kp, self.ki, self.i_max = kp, ki, i_max
        self.i = 0.0
    def reset(self):
        self.i = 0.0
    def update(self, err, dt, u_prev, u_min=0.0, u_max=1.0):
        dt = max(dt, 1e-3)
        # anti-windup: integrate only if not saturating against error
        u_prop = self.kp * err
        i_new = self.i + self.ki * err * dt
        i_new = max(-self.i_max, min(self.i_max, i_new))
        u = u_prop + i_new
        if (u >= u_max and err > 0) or (u <= u_min and err < 0):
            # don’t wind further in the wrong direction
            i_new = self.i
            u = self.kp * err + i_new
        self.i = i_new
        return max(u_min, min(u_max, u))

def ema(prev, new, tau, dt):
    if prev is None: return new
    alpha = dt / (tau + dt)
    return prev + alpha * (new - prev)

def slew(prev, target, rate_per_s, dt):
    if prev is None: return target
    max_step = rate_per_s * dt
    d = target - prev
    if d >  max_step: d =  max_step
    if d < -max_step: d = -max_step
    return prev + d

def run_path_follower(stop_event, status_cb=None):
    def set_status(msg):
        if status_cb:
            status_cb(msg)

    # ---- Connect FSDS for actuation only ----
    set_status("Connecting to FSDS…")
    client = fsds.FSDSClient()
    client.reset()
    client.confirmConnection()
    client.enableApiControl(True)

    # ---- Subscribe to localisation stream ----
    set_status(f"Waiting for localisation @ {LOC_HOST}:{LOC_PORT} …")
    loc = LocaliseClient(host=LOC_HOST, port=LOC_PORT)
    loc.start()

    ready = loc.wait_for_stream(min_samples=WARMUP_MIN_SAMPLES,
                                timeout=WARMUP_TIMEOUT_S,
                                max_age_s=STALE_MAX_AGE_S)
    if not ready:
        set_status("Still waiting for localisation… (car held)")
        # keep car safe while waiting forever
        while not loc.wait_for_stream(min_samples=3, timeout=1.0, max_age_s=STALE_MAX_AGE_S):
            cc = fsds.CarControls(); cc.throttle=0.0; cc.brake=SAFE_BRAKE; cc.steering=0.0
            client.setCarControls(cc)
            if stop_event.is_set(): return
    set_status("Localisation OK")

    # ---- Load path ----
    set_status("Loading pathpoints CSV…")
    route_x, route_y, route_s, route_len = build_route()

    # ---- Control loop state ----
    t_pred, cur_x, cur_y, yaw, speed = loc.get_state()
    cur_idx = int(np.argmin((route_x - cur_x)**2 + (route_y - cur_y)**2))

    last_t = time.perf_counter()
    v_meas_f = None
    v_cmd_prev = CRUISE_SPEED if CRUISE_MODE else 0.0
    throttle_cmd = 0.0
    steer_cmd = 0.0

    pi = PI(PI_KP, PI_KI, PI_I_MAX)

    set_status("Running…")

    try:
        while True:
            if stop_event.is_set():
                print("Stopped by user.")
                break

            fresh = loc.is_fresh(max_age_s=STALE_MAX_AGE_S)
            t_pred, cx, cy, yaw, v_meas = loc.get_state()

            now = time.perf_counter()
            dt = now - last_t
            last_t = now

            # stale → hold/brake
            if not fresh:
                cc = fsds.CarControls()
                cc.throttle, cc.brake, cc.steering = 0.0, SAFE_BRAKE, 0.0
                pi.reset()
                client.setCarControls(cc)
                set_status("Holding: localisation stale")
                time.sleep(0.02)
                continue

            # nearest path index
            cur_idx = local_closest_index((cx, cy), route_x, route_y, cur_idx, loop=ROUTE_IS_LOOP)

            # --- speed filtering
            v_meas_f = ema(v_meas_f, v_meas, SPEED_TAU, dt)

            # --- desired speed
            if CRUISE_MODE:
                v_target = CRUISE_SPEED
            else:
                v_target = CRUISE_SPEED  # keep simple for now

            # ramp target to limit jerk
            dv_max = V_CMD_RAMP_A * dt
            if v_target > v_cmd_prev + dv_max: v_cmd = v_cmd_prev + dv_max
            elif v_target < v_cmd_prev - dv_max: v_cmd = v_cmd_prev - dv_max
            else: v_cmd = v_target
            v_cmd_prev = v_cmd

            # --- PI speed control with deadband
            err = v_cmd - v_meas_f
            throttle_raw = 0.0
            brake_cmd = 0.0
            if err >= -SPEED_DEADBAND:
                # throttle only (brake off in deadband & positive side)
                throttle_raw = pi.update(err, dt, throttle_cmd, 0.0, 1.0)
                brake_cmd = 0.0
            else:
                # strong negative error → brake; reset PI to avoid windup
                pi.reset()
                throttle_raw = 0.0
                brake_cmd = min(1.0, BRAKE_GAIN * (-err))

            # throttle slew rate (separate up/down)
            if throttle_raw >= throttle_cmd:
                throttle_cmd = slew(throttle_cmd, throttle_raw, THROTTLE_SLEW_UP, dt)
            else:
                throttle_cmd = slew(throttle_cmd, throttle_raw, THROTTLE_SLEW_DN, dt)

            # --- steering (pure pursuit) + smoothing
            steer_pp, tgt_idx = pure_pursuit_steer((cx, cy), yaw, v_meas_f, route_x, route_y,
                                                   cur_idx, route_s, route_len, loop=ROUTE_IS_LOOP)
            steer_target = -STEER_GAIN * steer_pp  # sign as per your original
            # rate limit then EMA
            steer_cmd = slew(steer_cmd, steer_target, STEER_SLEW, dt)
            steer_cmd = ema(steer_cmd, steer_target, STEER_TAU, dt)
            steer_cmd = max(-1.0, min(1.0, steer_cmd))

            # --- send to sim
            cc = fsds.CarControls()
            cc.throttle = float(throttle_cmd)
            cc.brake    = float(brake_cmd)
            cc.steering = float(steer_cmd)
            client.setCarControls(cc)

            # exit condition (non-loop)
            if (not ROUTE_IS_LOOP) and (cur_idx >= len(route_x) - 1) and (v_meas_f < STOP_SPEED_THRESHOLD):
                print("Reached end of route and stopped. Exiting loop.")
                cc.handbrake = True
                client.setCarControls(cc)
                break

            time.sleep(0.02)

    finally:
        set_status("Idle")
        try: loc.stop()
        except: pass

# =========================================================
#                      SIMPLE TK GUI
# =========================================================
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("FSDS Path Follower (Localise-fed)")

        self.status_var = tk.StringVar(value="Idle")
        self.start_btn = ttk.Button(root, text="Start", command=self.on_start)
        self.stop_btn  = ttk.Button(root, text="Stop",  command=self.on_stop, state="disabled")
        self.quit_btn  = ttk.Button(root, text="Quit",  command=self.on_quit)

        self.status_lbl = ttk.Label(root, textvariable=self.status_var, anchor="w")

        self.start_btn.grid(row=0, column=0, padx=8, pady=8, sticky="ew")
        self.stop_btn.grid(row=0, column=1, padx=8, pady=8, sticky="ew")
        self.quit_btn.grid(row=0, column=2, padx=8, pady=8, sticky="ew")
        self.status_lbl.grid(row=1, column=0, columnspan=3, padx=8, pady=(0,8), sticky="ew")

        for i in range(3):
            root.grid_columnconfigure(i, weight=1)

        self.worker = None
        self.stop_event = None
        self.running = False

    def set_status(self, msg):
        self.root.after(0, self.status_var.set, msg)

    def on_start(self):
        if self.running:
            return
        self.stop_event = threading.Event()
        self.worker = threading.Thread(target=run_path_follower, args=(self.stop_event, self.set_status), daemon=True)
        self.worker.start()
        self.running = True
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.set_status("Starting…")

    def on_stop(self):
        if not self.running:
            return
        self.set_status("Stopping…")
        self.stop_event.set()
        def wait_join():
            if self.worker.is_alive():
                self.root.after(50, wait_join)
            else:
                self.running = False
                self.start_btn.config(state="normal")
                self.stop_btn.config(state="disabled")
                self.set_status("Idle")
        wait_join()

    def on_quit(self):
        if self.running:
            self.stop_event.set()
        self.root.after(200, self.root.destroy)

if __name__ == "__main__":
    root = tk.Tk()
    try:
        style = ttk.Style()
        if "vista" in style.theme_names():
            style.theme_use("vista")
    except Exception:
        pass
    App(root)
    root.mainloop()
