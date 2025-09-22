import os
import sys
import time
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from transforms.transform_1 import best_fit_transform
from transforms.transform_2 import icp
import json








# FSDS import / setup

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, repo_root)
import fsds

client = fsds.FSDSClient()
client.confirmConnection()
client.reset()
client.enableApiControl(True)

data_dir = os.path.join(os.path.dirname(__file__), 'data')
results_dir = os.path.join(data_dir, 'results')
os.makedirs(results_dir, exist_ok=True)

# # near start after results_dir is defined
# calib_file = os.path.join(results_dir, 'imu_calib.json')
# if os.path.exists(calib_file):
#     with open(calib_file,'r') as fh:
#         calib = json.load(fh)
#     B_AX = calib.get('b_ax', 0.0)
#     B_AY = calib.get('b_ay', 0.0)
#     B_WZ = calib.get('b_wz', 0.0)
#     print("Loaded IMU calibration:", calib)
# else:
#     B_AX = B_AY = B_WZ = 0.0
# --- Online bias estimator state (place near other initializations) ---
# Load previous calibration if available (optional)


calib_file = os.path.join(results_dir, 'imu_calib.json')
if os.path.exists(calib_file):
    try:
        with open(calib_file,'r') as fh:
            _calib = json.load(fh)
        B_AX = float(_calib.get('b_ax', 0.0))
        B_AY = float(_calib.get('b_ay', 0.0))
        B_WZ = float(_calib.get('b_wz', 0.0))
        print("Loaded IMU calibration for online estimator:", _calib)
    except Exception:
        B_AX = B_AY = B_WZ = 0.0
else:
    B_AX = B_AY = B_WZ = 0.0

# Gains for online updates (tune these: smaller = slower, larger = more aggressive)
GAIN_GYRO = 0.5   # rad/s per rad of heading error per second (affects bias_wz update)
GAIN_ACCEL_V = 0.5 # m/s^2 per (m/s) velocity error per second (affects accel bias)
GAIN_ZUPT = 2.0   # stronger gain during zero-speed (m/s^2 per m/s of vel error)
SPEED_ZUPT_THRESH = 0.15  # m/s threshold to consider vehicle stopped for ZUPT
BIAS_MAX = 1.0    # clamp biases to reasonable bounds (m/s^2 or rad/s)








# Config - Stress tests

USE_GPS         = False          # Flip to False for A/B: LiDAR+IMU only
USE_LIDAR       = False        # Flip to False for A/B: GPS+IMU only
GPS_DELAY_SEC   = 0.00           # e.g. 0.10 or 0.20 to simulate latency
LIDAR_DROP_PROB = 0.00           # e.g. 0.30 to simulate sparse/occluded frames

# Association gate - Euclidean distance
ASSOC_GATE            = 1.5
MIN_PAIRS_FOR_UPDATE  = 2

Q = np.diag([0.01774506, 0.01774506, np.deg2rad(0.00150152),0.19716737])


R_GPS = np.diag([1.0, 1.0]) 


# Load static map

map_pts = pd.read_csv(os.path.join(data_dir, 'combined_coordinates.csv'))[['x','y']].to_numpy()
print(f"{map_pts.shape[0]} map cones loaded.")
kd = cKDTree(map_pts)


# State: [x, y, theta, v]

x_est = np.zeros(4)
P_est = np.diag([1.0, 1.0, np.deg2rad(10)**2, 1.0])

# IMU-only dead-reckoning state [x, y, theta, v]
imu_x = np.zeros(4)
imu_x[:2] = x_est[:2]   # start from same initial position
imu_x[2]  = x_est[2]
imu_x[3]  = x_est[3]

imu_xs, imu_ys = [], []
g_world = np.array([0.0, 0.0, -9.81])   # m/s²


# Utilities for tests

def wrap_angle(a: float) -> float:
    return (a + np.pi) % (2*np.pi) - np.pi

def yaw_from_quat(q) -> float:
    qw, qx, qy, qz = float(q.w_val), float(q.x_val), float(q.y_val), float(q.z_val)
    
    siny_cosp = 2.0*(qw*qz + qx*qy)
    cosy_cosp = 1.0 - 2.0*(qy*qy + qz*qz)
    return math.atan2(siny_cosp, cosy_cosp)

def rmse(a: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(a))))

def compute_ate_rpe(est_xy: np.ndarray, gt_xy: np.ndarray, dt: float, rpe_dt: float = 1.0):
  
    n = min(len(est_xy), len(gt_xy))
    if n < 2:
        return np.nan, np.nan

    est = est_xy[:n]
    gt  = gt_xy[:n]

    # ATE on position
    ate_rmse = rmse(np.linalg.norm(est - gt, axis=1))

    # RPE  
    k = int(round(rpe_dt / max(dt, 1e-6)))
    if k >= 1 and n > k:
        rel_err = []
        for i in range(n - k):
            de = est[i+k] - est[i]
            dg = gt[i+k]  - gt[i]
            rel_err.append(np.linalg.norm(de - dg))
        rpe_rmse = rmse(np.array(rel_err))
    else:
        rpe_rmse = np.nan

    return ate_rmse, rpe_rmse


# EKF Predict - IMU

def ekf_predict(x, P, imu, v_meas, dt):
    a     = float(imu.linear_acceleration.x_val)
    omega = float(imu.angular_velocity.z_val)

    # Blend measured speed with predicted speed to avoid oscillations
    v_blend = 0.8 * v_meas + 0.2 * (x[3] + a*dt) if not np.isnan(v_meas) else (x[3] + a*dt)

    th = x[2]
    x_pred = x.copy()
    x_pred[0] += v_blend*np.cos(th)*dt
    x_pred[1] += v_blend*np.sin(th)*dt
    x_pred[2]  = wrap_angle(th + omega*dt)
    x_pred[3]  = v_blend

    F = np.eye(4)
    F[0,2] = -v_blend*np.sin(th)*dt
    F[0,3] =  np.cos(th)*dt
    F[1,2] =  v_blend*np.cos(th)*dt
    F[1,3] =  np.sin(th)*dt

    P_pred = F @ P @ F.T + Q
    return x_pred, P_pred


# LiDAR clustering from example

def find_cones():
    lid = client.getLidarData(lidar_name='Lidar')
    pts = np.array(lid.point_cloud, dtype=np.float32).reshape(-1,3)[:,:2]
    if len(pts)==0: 
        return []
    clusters = []
    current = [pts[0]]
    for p_prev, p in zip(pts, pts[1:]):
        if np.linalg.norm(p-p_prev) < 0.1:
            current.append(p)
        else:
            if current:
                avg = np.mean(current,axis=0)
                if np.linalg.norm(avg) < 7.0:
                    clusters.append({'x':float(avg[0]),'y':float(avg[1])})
            current = [p]
    return clusters


# EKF Update - LiDAR 

def ekf_update_scan(x_pred, P_pred, cones):
    if not cones:
        return (x_pred, P_pred, 
                np.empty((0,2)), np.empty((0,2)), 
                np.nan,       
                0, 0,          # n_total, n_gated
                0.0, 0.0)      

    th = x_pred[2]
    R_pred = np.array([[ np.cos(th), -np.sin(th)],
                       [ np.sin(th),  np.cos(th)]])
    local_pts = np.array([[c['x'], c['y']] for c in cones])
    glob_pts  = (R_pred @ local_pts.T).T + x_pred[:2]

    # Nearest-neighbor association vs map
    dists, idxs = kd.query(glob_pts)
    matched_map = map_pts[idxs]

    # Gating
    mask = (dists <= ASSOC_GATE)
    n_total = int(glob_pts.shape[0])
    n_gated = int(np.count_nonzero(mask))

    if n_gated < MIN_PAIRS_FOR_UPDATE:
        # skip update but return stats so we can see it
        return (x_pred, P_pred, 
                glob_pts, matched_map, 
                np.nan, n_total, n_gated, 
                0.0, 0.0)

    gp_in  = glob_pts[mask]
    mp_in  = matched_map[mask]

    # Initial best-fit 
    R_map2glob, t_map2glob = best_fit_transform(mp_in, gp_in)
    R_corr = R_map2glob.T
    t_corr = - R_corr.dot(t_map2glob)

    # ICP refine if enough pairs
    try:
        if gp_in.shape[0] >= MIN_PAIRS_FOR_UPDATE:
            R_icp, t_icp, _ = icp(mp_in, gp_in)
            R_corr = R_icp.T
            t_corr = - R_corr.dot(t_icp)
    except Exception:
        pass
    
    dtheta = np.arctan2(R_corr[1,0], R_corr[0,0])
    z_scan = np.array([
        x_pred[0] + t_corr[0],
        x_pred[1] + t_corr[1],
        wrap_angle(x_pred[2] + dtheta)
    ])

    # Data-driven R for LiDAR 
    fitted = (R_corr @ gp_in.T).T + t_corr
    errs   = mp_in - fitted
    if errs.shape[0] >= 2:
        var_xy = np.var(errs, axis=0)
    else:
        var_xy = np.array([0.05**2, 0.05**2])
    var_xy = np.maximum(var_xy, np.array([0.03**2, 0.03**2]))   # ≥3 cm
    var_th = max(np.deg2rad(0.7)**2, np.deg2rad(0.5)**2)
    R_scan = np.diag([var_xy[0], var_xy[1], var_th])

    H = np.hstack([np.eye(3), np.zeros((3,1))])  # 3×4
    y = z_scan - H @ x_pred
    y[2] = wrap_angle(y[2])

    S = H @ P_pred @ H.T + R_scan
    K = P_pred @ H.T @ np.linalg.inv(S)

    try:
        nis_scan = float(y.T @ np.linalg.inv(S) @ y)
    except np.linalg.LinAlgError:
        nis_scan = np.nan

    # Joseph form 
    I = np.eye(4)
    x_upd = x_pred + K @ y
    x_upd[2] = wrap_angle(x_upd[2])
    P_upd = (I - K @ H) @ P_pred @ (I - K @ H).T + K @ R_scan @ K.T

    delta_traceP = float(np.trace(P_pred) - np.trace(P_upd))
    K_fro        = float(np.linalg.norm(K, ord='fro'))

    return (x_upd, P_upd, 
            glob_pts, matched_map, 
            nis_scan, n_total, n_gated, 
            delta_traceP, K_fro)


# EKF Update: GPS

def ekf_update_gps(x_pred, P_pred, car_state):
    z     = np.array([car_state.position.x_val, car_state.position.y_val])
    H     = np.array([[1,0,0,0],[0,1,0,0]])

    y     = z - H @ x_pred
    S     = H @ P_pred @ H.T + R_GPS
    K     = P_pred @ H.T @ np.linalg.inv(S)
    I     = np.eye(4)


    try:
        nis_gps = float(y.T @ np.linalg.inv(S) @ y)
    except np.linalg.LinAlgError:
        nis_gps = np.nan

    x_upd = x_pred + K @ y
    x_upd[2] = wrap_angle(x_upd[2])
    P_upd = (I - K @ H) @ P_pred @ (I - K @ H).T + K @ R_GPS @ K.T
    return x_upd, P_upd, nis_gps


# Logs

imu_logs        = []
gps_logs        = []
lidar_logs      = []
gt_logs         = []  # ground-truth from sim
consistency_logs= []  # NIS, trace(P), etc
health_logs     = []  # eigenvalues, counts
lidar_use_logs  = []  # proves LiDAR usage
sensor_stats    = {'gps': {'dtrP': [], 'count':0},
                   'lidar': {'dtrP': [], 'count':0}}
est_xs, est_ys, est_yaws = [], [], []

# Optional GPS delay buffer
gps_buf = []
gt_xs, gt_ys = [], []



# Visualizer 
client.enableApiControl(False)
plt.ion()
fig, ax = plt.subplots(figsize=(6,6))
ax.set_aspect('equal','box')

# Plot map cones
ax.scatter(map_pts[:,0], map_pts[:,1], c='lightgray', s=20, label='Map Cones')
scan_sc     = ax.scatter([], [], c='green', s=5, alpha=0.6, label='Clusters')
traj_line   = ax.plot([], [], c='red',     linewidth=2, label='EKF Trajectory')[0]
imu_line    = ax.plot([], [], c='blue', linestyle='--', linewidth=1.5, label='IMU Trajectory')[0]
gt_line     = ax.plot([], [], c='black', linewidth=2, alpha=0.7, label='Ground Truth')[0]
assoc_lines = []

ax.legend(loc='upper right')
ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
ax.set_title('EKF-SLAM')

xmin, xmax = map_pts[:,0].min()-1, map_pts[:,0].max()+1
ymin, ymax = map_pts[:,1].min()-1, map_pts[:,1].max()+1
ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)

# Display transform 
disp_ready = False
R_disp = np.array([[0.0, -1.0],
                   [1.0,  0.0]])
t_disp = np.zeros(2)

def apply_disp(xy):
    xy = np.asarray(xy, dtype=float)
    if xy.ndim == 1:
        return R_disp @ xy + t_disp
    else:
        return (R_disp @ xy.T).T + t_disp


# Initialize from car state

target_dt = 1.0/20.0
last_t = time.time()

cs0 = client.getCarState().kinematics_estimated
x_est[:2] = [cs0.position.x_val, cs0.position.y_val]
x_est[2]  = yaw_from_quat(cs0.orientation)
x_est[3]  = float(np.hypot(cs0.linear_velocity.x_val, cs0.linear_velocity.y_val))

print("Visualizer running.")
try:
    while True:
        t_now = time.time()
        dt = max(1e-3, min(0.2, t_now - last_t))
        last_t = t_now


        imu = client.getImuData("Imu")
        car_state = client.getCarState().kinematics_estimated
        v_meas = float(np.hypot(car_state.linear_velocity.x_val, car_state.linear_velocity.y_val))

        # Logs 
        imu_logs.append({
          'time':t_now,
          'ax': imu.linear_acceleration.x_val,
          'ay': imu.linear_acceleration.y_val,
          'az': imu.linear_acceleration.z_val,
          'ωx': imu.angular_velocity.x_val,
          'ωy': imu.angular_velocity.y_val,
          'ωz': imu.angular_velocity.z_val
        })

        gpsd = client.getGpsData("Gps")
        gps_logs.append({
          'time':t_now,
          'lat': gpsd.gnss.geo_point.latitude,
          'lon': gpsd.gnss.geo_point.longitude,
          'alt': gpsd.gnss.geo_point.altitude
        })

        # Ground truth pose from sim 
        gt_logs.append({
          'time': t_now,
          'gt_x': float(car_state.position.x_val),
          'gt_y': float(car_state.position.y_val),
          'gt_yaw': yaw_from_quat(car_state.orientation)
        })
        gt_xs.append(float(car_state.position.x_val))
        gt_ys.append(float(car_state.position.y_val))
        x_est, P_est = ekf_predict(x_est, P_est, imu, v_meas, dt)
        
        
        # IMU-only integration (dead-reckoning)
        a     = float(imu.linear_acceleration.x_val)
        omega = float(imu.angular_velocity.z_val)

        # Update velocity & heading
        imu_x[3] += a * dt
        imu_x[2] = wrap_angle(imu_x[2] + omega * dt)

        # Update position
        imu_x[0] += imu_x[3] * np.cos(imu_x[2]) * dt
        imu_x[1] += imu_x[3] * np.sin(imu_x[2]) * dt

        imu_xs.append(imu_x[0]); imu_ys.append(imu_x[1])
           

        # Default stats if no update
        nis_scan = np.nan
        n_total = 0
        n_gated = 0
        dtrP_lidar = 0.0
        Kfro_lidar = 0.0

        if USE_LIDAR:
            cones = find_cones()
            # Random frame drop to simulate sparsity/occlusion
            use_this_frame = (np.random.rand() > LIDAR_DROP_PROB)
            if cones and use_this_frame:
                # Save raw clusters 
                th = x_est[2]
                R_pred = np.array([[ np.cos(th), -np.sin(th)],
                                   [ np.sin(th),  np.cos(th)]])
                local_pts = np.array([[c['x'], c['y']] for c in cones])
                glob_pts_raw  = (R_pred @ local_pts.T).T + x_est[:2]
                tt = time.time()
                for lp, gp in zip(local_pts, glob_pts_raw):
                    lidar_logs.append({
                      'time':     tt,
                      'local_x':  float(lp[0]), 'local_y':  float(lp[1]),
                      'global_x': float(gp[0]), 'global_y': float(gp[1])
                    })
                # Update 
                tb = float(np.trace(P_est))
                (x_est, P_est, glob_pts, matched_map, 
                 nis_scan, n_total, n_gated, dtrP_lidar, Kfro_lidar) = ekf_update_scan(x_est, P_est, cones)
                sensor_stats['lidar']['dtrP'].append(dtrP_lidar)
                sensor_stats['lidar']['count'] += (1 if n_gated >= MIN_PAIRS_FOR_UPDATE else 0)
            else:
                glob_pts    = np.empty((0,2))
                matched_map = np.empty((0,2))
        else:
            glob_pts    = np.empty((0,2))
            matched_map = np.empty((0,2))

        nis_gps = np.nan
        if USE_GPS:
            gps_buf.append((t_now, car_state))
            # release buffered measurements older than delay
            while gps_buf and (t_now - gps_buf[0][0]) >= GPS_DELAY_SEC:
                _, delayed_cs = gps_buf.pop(0)
                tb = float(np.trace(P_est))
                x_est, P_est, nis_gps = ekf_update_gps(x_est, P_est, delayed_cs)
                sensor_stats['gps']['dtrP'].append(tb - float(np.trace(P_est)))
                sensor_stats['gps']['count'] += 1

        consistency_logs.append({
          'time': t_now,
          'nis_scan': float(nis_scan) if not np.isnan(nis_scan) else np.nan,
          'nis_gps':  float(nis_gps) if not np.isnan(nis_gps) else np.nan,
          'traceP':   float(np.trace(P_est))
        })

        eigP = np.linalg.eigvals(P_est)
        min_eigP = float(np.min(np.real(eigP)))
        health_logs.append({
          'time': t_now,
          'min_eigP': min_eigP,
          'traceP': float(np.trace(P_est)),
          'n_lidar_pairs': int(n_gated)
        })

        lidar_use_logs.append({
          'time': t_now,
          'n_total': int(n_total),
          'n_gated': int(n_gated),
          'nis_scan': float(nis_scan) if not np.isnan(nis_scan) else np.nan,
          'delta_traceP': float(dtrP_lidar),
          'K_fro': float(Kfro_lidar)
        })

        # Keep EKF trajectory 
        est_xs.append(x_est[0]); est_ys.append(x_est[1]); est_yaws.append(x_est[2])   

        traj_raw = np.column_stack([est_xs, est_ys])
        if not disp_ready and traj_raw.shape[0] >= 1:
            first_raw = traj_raw[0]
            desired   = np.array([0.0, -15.0])
            t_disp[:] = desired - (R_disp @ first_raw)
            disp_ready = True

        
        used_now = (n_gated >= MIN_PAIRS_FOR_UPDATE)

        # ---------------- Online bias correction (run each loop) ----------------


        # imu_pos and imu_theta are the IMU-only pose we just integrated
        imu_pos = np.array([imu_x[0], imu_x[1]])
        imu_theta = imu_x[2]
        ekf_pos = x_est[:2].copy()
        ekf_theta = x_est[2]

        # Residuals (IMU - EKF)
        pos_res = imu_pos - ekf_pos
        theta_res = wrap_angle(imu_theta - ekf_theta)

        # Update gyro bias using heading residual when a trusted correction happened
        # Use proportional update scaled by dt to keep units sane
        if (n_gated >= MIN_PAIRS_FOR_UPDATE) or (USE_GPS and gps_updates>0 and sensor_stats['gps']['count']>0):
            # If EKF used an update this frame, treat EKF as ground truth-ish for bias correction
            # Update bias on omega (we want to remove bias so that imu_theta matches ekf_theta)
            d_b_wz = - GAIN_GYRO * (theta_res) / max(1e-6, dt)
            if pos_body[1]<0:
                d_b_wz *=-1
            B_WZ += d_b_wz * dt  # multiply by dt to make step size proportional to dt
            # clamp
            B_WZ = float(np.clip(B_WZ, -BIAS_MAX, BIAS_MAX))

            # For accel bias, update based on velocity difference projected along heading.
            # Compute imu scalar forward speed and ekf speed
            v_imu = imu_x[3]
            v_ekf = x_est[3] if not np.isnan(x_est[3]) else v_imu
            vel_err = (v_imu - v_ekf)

            # d_b_ax approximately aims to correct forward accel bias (projected to body x)
            d_b_ax = -GAIN_ACCEL_V * vel_err / max(1e-6, dt)
            B_AX += d_b_ax * dt
            B_AX = float(np.clip(B_AX, -BIAS_MAX, BIAS_MAX))

            # Lateral accel bias correction using lateral position residual projected to body y
            # Project pos_res into body frame to see lateral offset
            Rinv = np.array([[np.cos(imu_theta), np.sin(imu_theta)],
                            [-np.sin(imu_theta), np.cos(imu_theta)]])  # rotation global->body
            pos_body = Rinv.dot(pos_res)
            # pos_body[1] is lateral offset; convert to acceleration bias update

            d_b_ay = -GAIN_ACCEL_V * (pos_body[1]) / max(1e-6, dt)
            if pos_body[1] <0:
                d_b_ax*=-1
            B_AY += d_b_ay * dt
            B_AY = float(np.clip(B_AY, -BIAS_MAX, BIAS_MAX))

        # ZUPT-like correction: if measured speed is near zero (car stopped), force velocity -> 0
        measured_speed = v_meas
        if measured_speed < SPEED_ZUPT_THRESH:
            # correct accel bias to drive imu_x[3] toward zero
            vel_err = imu_x[3]  # want this to be 0
            d_b_ax = - GAIN_ZUPT * vel_err / max(1e-6, dt)
            B_AX += d_b_ax * dt
            B_AX = float(np.clip(B_AX, -BIAS_MAX, BIAS_MAX))
        # -----------------------------------------------------------------------


        traj_line.set_color('red' if used_now else 'gray')

        if traj_raw.size:
            traj_disp = apply_disp(traj_raw)
            traj_line.set_data(traj_disp[:,0], traj_disp[:,1])
        else:
            traj_line.set_data([], [])
        

        # IMU-only trajectory
        imu_traj_raw = np.column_stack([imu_xs, imu_ys])
        if imu_traj_raw.size:
            imu_traj_disp = apply_disp(imu_traj_raw)
            imu_line.set_data(imu_traj_disp[:,0], imu_traj_disp[:,1])
        else:
            imu_line.set_data([], [])


        # Ground truth trajectory
        gt_traj_raw = np.column_stack([gt_xs, gt_ys])
        if gt_traj_raw.size:
            gt_traj_disp = apply_disp(gt_traj_raw)
            gt_line.set_data(gt_traj_disp[:,0], gt_traj_disp[:,1])
        else:
            gt_line.set_data([], [])

        # Plot LiDAR clusters and assoc lines
        if glob_pts.size:
            gp_disp = apply_disp(glob_pts)
            scan_sc.set_offsets(gp_disp)
        else:
            scan_sc.set_offsets(np.empty((0,2)))

        for ln in assoc_lines: ln.remove()
        assoc_lines.clear()
        if glob_pts.size and matched_map.size:
            gp_disp = apply_disp(glob_pts)
            mp_disp = apply_disp(matched_map)
            for (gx,gy),(mx,my) in zip(gp_disp, mp_disp):
                ln, = ax.plot([gx,mx],[gy,my],'y--',linewidth=1)
                assoc_lines.append(ln)

        fig.canvas.draw(); fig.canvas.flush_events()

        # Pace loop
        time.sleep(max(0, target_dt - (time.time()-t_now)))

except KeyboardInterrupt:
    print("\nVisualization stopped. Computing metrics and saving logs...")
    plt.ioff(); plt.show()


    # Save log 

    pd.DataFrame(imu_logs).to_csv(os.path.join(results_dir,'imu_log.csv'),   index=False)
    pd.DataFrame(gps_logs).to_csv(os.path.join(results_dir,'gps_log.csv'),   index=False)
    pd.DataFrame(lidar_logs).to_csv(os.path.join(results_dir,'lidar_log.csv'),index=False)
    pd.DataFrame(consistency_logs).to_csv(os.path.join(results_dir,'consistency_log.csv'), index=False)
    pd.DataFrame(health_logs).to_csv(os.path.join(results_dir,'health_log.csv'), index=False)
    pd.DataFrame(lidar_use_logs).to_csv(os.path.join(results_dir,'lidar_use_log.csv'), index=False)

 
    # Compute ATE/RPE vs ground truth

    est_xy = np.column_stack([est_xs, est_ys])
    gt_xy  = np.column_stack([[g['gt_x'] for g in gt_logs],
                              [g['gt_y'] for g in gt_logs]])
    ate_rmse, rpe_rmse = compute_ate_rpe(est_xy, gt_xy, dt=target_dt, rpe_dt=1.0)




     # --- IMU-only error vs ground truth ---
    imu_xy = np.column_stack([imu_xs, imu_ys])
    gt_xy  = np.column_stack([[g['gt_x'] for g in gt_logs],
                              [g['gt_y'] for g in gt_logs]])


    # Align lengths
    n = min(len(imu_xy), len(gt_xy))
    imu_xy = imu_xy[:n]
    gt_xy  = gt_xy[:n]

    imu_rmse = rmse(np.linalg.norm(imu_xy - gt_xy, axis=1))
    print(f"IMU-only trajectory RMSE vs GT: {imu_rmse:.3f} m")

    
    # --- BEGIN: IMU calibration (bias estimation) ------------------------------

    try:
        from scipy.optimize import minimize
    except Exception:
        minimize = None

    def simulate_imu_traj(imu_logs, biases, dt_default=1.0/20.0):
        """
        Simulate 2D IMU dead-reckoning using logs.
        imu_logs: list of dicts with keys 'time', 'ax','ay','az','ωx','ωy','ωz'
        biases: [b_ax, b_ay, b_omega_z]
        Returns Nx2 array of positions (x,y)
        """
        b_ax, b_ay, b_omega = biases
        xs, ys = [], []
        x = np.zeros(4)   # [x,y,theta,v]
        last_t = imu_logs[0]['time'] if imu_logs else 0.0
        for rec in imu_logs:
            t = rec['time']
            dt = max(1e-6, min(0.2, t - last_t)) if 'time' in rec else dt_default
            last_t = t

            # ---------------- IMU-only integration (bias-corrected) -----------------
            # Read raw IMU
            raw_ax = float(imu.linear_acceleration.x_val)
            raw_ay = float(imu.linear_acceleration.y_val)
            raw_wz = float(imu.angular_velocity.z_val)

            # Apply online bias correction
            ax = raw_ax - B_AX
            ay = raw_ay - B_AY
            omega = raw_wz - B_WZ

            # integrate IMU-only state
            imu_x[2] = wrap_angle(imu_x[2] + omega * dt)   # heading
            # rotate accel into world frame
            Rimu = np.array([[np.cos(imu_x[2]), -np.sin(imu_x[2])],
                            [np.sin(imu_x[2]),  np.cos(imu_x[2])]])
            a_world = Rimu.dot(np.array([ax, ay]))
            # project acceleration along heading to increment scalar speed
            a_proj = a_world[0]*np.cos(imu_x[2]) + a_world[1]*np.sin(imu_x[2])
            imu_x[3] += a_proj * dt
            # integrate position
            imu_x[0] += imu_x[3] * np.cos(imu_x[2]) * dt
            imu_x[1] += imu_x[3] * np.sin(imu_x[2]) * dt

            imu_xs.append(imu_x[0]); imu_ys.append(imu_x[1])
            # -----------------------------------------------------------------------


        return np.column_stack([xs, ys])

    def rmse_traj(pred_xy, gt_xy):
        n = min(len(pred_xy), len(gt_xy))
        if n < 1: return np.nan
        return rmse(np.linalg.norm(pred_xy[:n] - gt_xy[:n], axis=1))

    # Prepare data arrays
    imu_arr = imu_logs.copy()
    gt_xy = np.column_stack([[g['gt_x'] for g in gt_logs],[g['gt_y'] for g in gt_logs]])
    if len(imu_arr) < 10 or gt_xy.shape[1] == 0:
        print("Not enough IMU/GT data for calibration.")
    else:
        # baseline: raw IMU rmse (should already be computed above, but recompute for clarity)
        imu_raw = np.column_stack([imu_xs, imu_ys])
        n = min(len(imu_raw), len(gt_xy))
        imu_raw = imu_raw[:n]; gt_xy_trim = gt_xy[:n]
        raw_rmse = rmse(np.linalg.norm(imu_raw - gt_xy_trim, axis=1))
        print(f"\n[Calib] IMU raw RMSE: {raw_rmse:.4f} m")

        # Objective to minimize: RMSE between simulated imu traj (with biases) and GT
        def objective(bias_vec):
            sim_xy = simulate_imu_traj(imu_arr, bias_vec)
            # align lengths with GT (use earliest points)
            n = min(len(sim_xy), len(gt_xy))
            if n < 2: return 1e6
            return float(rmse(np.linalg.norm(sim_xy[:n] - gt_xy[:n], axis=1)))

        # initial guess: no bias
        x0 = np.array([0.0, 0.0, 0.0])

        best = None
        if minimize is not None:
            try:
                res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter':300, 'xatol':1e-6, 'fatol':1e-6})
                if res.success:
                    best = res.x
                else:
                    best = res.x  # still use result even if not flagged success
            except Exception as e:
                print("[Calib] scipy.minimize failed:", e)
                best = None

        # fallback coarse grid search if optimizer not available / failed
        if best is None:
            print("[Calib] Using fallback grid search (coarse).")
            best_val = 1e12
            best = x0.copy()
            # try small biases in reasonable ranges
            ax_range = np.linspace(-0.5, 0.5, 9)   # m/s^2
            ay_range = np.linspace(-0.5, 0.5, 9)
            wz_range = np.linspace(-0.5, 0.5, 9)   # rad/s
            for bax in ax_range:
                for bay in ay_range:
                    for bw in wz_range:
                        val = objective([bax, bay, bw])
                        if val < best_val:
                            best_val = val
                            best = np.array([bax, bay, bw])
            print(f"[Calib] grid best val {best_val:.4f}")

        b_ax, b_ay, b_wz = float(best[0]), float(best[1]), float(best[2])
        print(f"[Calib] Estimated biases -> a_x: {b_ax:.6f} m/s^2, a_y: {b_ay:.6f} m/s^2, ω_z: {b_wz:.6f} rad/s")

        # Simulate corrected trajectory and compute RMSE
        sim_corr = simulate_imu_traj(imu_arr, [b_ax, b_ay, b_wz])
        n = min(len(sim_corr), len(gt_xy))
        sim_corr = sim_corr[:n]; gt_xy_trim = gt_xy[:n]
        corr_rmse = rmse(np.linalg.norm(sim_corr - gt_xy_trim, axis=1))
        print(f"[Calib] Corrected IMU RMSE: {corr_rmse:.4f} m  (was {raw_rmse:.4f} m)")

        # Save calibration to disk for reuse
        calib = {'b_ax': b_ax, 'b_ay': b_ay, 'b_wz': b_wz, 'raw_rmse': raw_rmse, 'corr_rmse': corr_rmse}
        with open(os.path.join(results_dir, 'imu_calib.json'), 'w') as fh:
            json.dump(calib, fh, indent=2)

        # Save corrected IMU-only trajectory to CSV for plotting / replay
        sim_df = pd.DataFrame(sim_corr, columns=['x','y'])
        sim_df.to_csv(os.path.join(results_dir, 'imu_corrected_traj.csv'), index=False)

        # Optionally update process noise Q slightly smaller since we've removed bias (not required)
        # We reduce the velocity/position process terms proportionally to RMSE improvement
        try:
            improvement = max(0.1, corr_rmse / (raw_rmse + 1e-9))
            Q_scale = improvement  # <1 if improved
            Q = np.diag([
                (0.03 * Q_scale)**2,
                (0.03 * Q_scale)**2,
                (np.deg2rad(0.5) * Q_scale)**2,
                (0.10 * Q_scale)**2
            ])
            print(f"[Calib] Auto-updated process noise Q with scale {Q_scale:.3f}")
            print(Q)
        except Exception:
            pass

    # --- END: IMU calibration ----------------------------------------------

        # Save online biases back to imu_calib.json
        try:
            calib = {'b_ax': B_AX, 'b_ay': B_AY, 'b_wz': B_WZ}
            with open(os.path.join(results_dir, 'imu_calib.json'), 'w') as fh:
                json.dump(calib, fh, indent=2)
            print("Saved online estimated biases:", calib)
        except Exception as e:
            print("Failed saving calib:", e)

        

        



        # LiDAR attribution summary

        lidar_df = pd.DataFrame(lidar_use_logs)
        if not lidar_df.empty:
            used_mask   = lidar_df['n_gated'] >= MIN_PAIRS_FOR_UPDATE
            used_pct    = 100.0 * float(used_mask.mean())
            median_pairs= float(lidar_df.loc[used_mask, 'n_gated'].median()) if used_mask.any() else 0
            median_dtrP = float(lidar_df.loc[used_mask, 'delta_traceP'].median()) if used_mask.any() else 0.0
            median_Kfro = float(lidar_df.loc[used_mask, 'K_fro'].median()) if used_mask.any() else 0.0
        else:
            used_pct = median_pairs = median_dtrP = median_Kfro = 0.0


        # Per-sensor trace(P) stats

        def _med(arr): 
            arr = np.asarray(arr)
            return float(np.median(arr)) if arr.size else 0.0

        gps_updates   = sensor_stats['gps']['count']
        lidar_updates = sensor_stats['lidar']['count']
        gps_med_dtrP   = _med(sensor_stats['gps']['dtrP'])
        lidar_med_dtrP = _med(sensor_stats['lidar']['dtrP'])

       
       
        # --- Auto-tune IMU covariance Q ---
    # Target RMSE (depends on your desired accuracy)
    target_rmse = 1.0  # meters over full run

    scale_factor = (imu_rmse / target_rmse)*0.1 if target_rmse > 0 else 1.0
    scale_factor = max(0.1, min(scale_factor, 10.0))  # keep sane bounds

    Q = np.diag([
        (0.03 * scale_factor)**2,            # x noise
        (0.03 * scale_factor)**2,            # y noise
        (np.deg2rad(0.5) * scale_factor)**2, # yaw noise
        (0.10 * scale_factor)**2             # velocity noise
    ])

    print("\nUpdated process noise covariance Q:")
    print(Q)




    print("\n===== RUN SUMMARY =====")
    print(f"ATE (pos) RMSE: {ate_rmse:.3f} m | RPE(1s) RMSE: {rpe_rmse:.3f} m")
    print(f"LiDAR frames with usable update (>= {MIN_PAIRS_FOR_UPDATE} pairs): {used_pct:.1f}%")
    print(f"   Median gated pairs (used frames): {median_pairs}")
    print(f"   Median Δtrace(P) from LiDAR (used frames): {median_dtrP:.4f}  (>0 → LiDAR tightened P)")
    print(f"GPS updates applied:   {gps_updates}, median Δtrace(P) per update: {gps_med_dtrP:.4f}")
    print(f"LiDAR updates applied: {lidar_updates}, median Δtrace(P) per update: {lidar_med_dtrP:.4f}")
    print("CSV logs saved in:", results_dir)
