# lidar_clusters_3d.py
# Real-time LiDAR 3D viewer from car perspective (no top-down, no Matplotlib).
# Clusters points in XY and renders with a virtual pinhole camera.

import time, math, random
from typing import List
import numpy as np
import cv2
import fsds

VEHICLE = "FSCar"
LIDAR   = "Lidar"

# Viewer canvas
WIN_W, WIN_H = 1000, 700
FOV_DEG = 75.0
FX = (WIN_W/2) / math.tan(math.radians(FOV_DEG/2))
FY = FX
CX, CY = WIN_W/2, WIN_H*0.55  # a little horizon offset

# Filters & clustering defaults
Z_MIN, Z_MAX = -0.2, 1.5   # meters (keep cone-ish band)
R_MAX = 45.0               # max range in XY
MAX_POINTS = 120000

EPS_INIT = 0.28            # m
MIN_PTS_INIT = 8

PALETTE = [(255, 128, 128), (128, 255, 128), (128, 128, 255),
           (255, 200,  80), (180, 180, 180), (160, 80, 255),
           (80, 220, 220),  (220, 80, 160),  (120, 240, 120)]

def get_lidar_points(client):
    data = client.getLidarData(lidar_name=LIDAR, vehicle_name=VEHICLE)
    if data is None or data.point_cloud is None:
        return np.empty((0,3), float), 0.0
    arr = np.asarray(data.point_cloud, dtype=np.float32)
    t = float(data.time_stamp) * 1e-9 if data.time_stamp > 1e12 else float(data.time_stamp)
    if arr.size == 0 or (arr.size % 3) != 0:
        return np.empty((0,3), float), t
    pts = arr.reshape(-1,3).astype(np.float32)
    return pts, t

def filter_body_points(pb: np.ndarray, use_ground_filter=True):
    if pb.size == 0:
        return pb
    r = np.hypot(pb[:,0], pb[:,1])
    m = (r <= R_MAX)
    if use_ground_filter:
        m &= (pb[:,2] >= Z_MIN) & (pb[:,2] <= Z_MAX)
    out = pb[m]
    if out.shape[0] > MAX_POINTS:
        step = int(math.ceil(out.shape[0] / MAX_POINTS))
        out = out[::step]
    return out

def cluster_xy(points: np.ndarray, eps: float, min_pts: int) -> List[np.ndarray]:
    """
    Simple O(N^2) Euclidean clustering in XY.
    Returns list of index arrays (one per cluster).
    """
    N = points.shape[0]
    if N == 0: return []
    visited = np.zeros(N, bool)
    clusters = []
    # use XY only
    xy = points[:, :2]
    for i in range(N):
        if visited[i]: continue
        q = [i]; visited[i] = True; comp = [i]
        while q:
            j = q.pop()
            d2 = np.sum((xy - xy[j])**2, axis=1)
            neigh = np.where(d2 <= eps*eps)[0]
            for k in neigh:
                if not visited[k]:
                    visited[k] = True
                    q.append(k)
                    comp.append(k)
        if len(comp) >= min_pts:
            clusters.append(np.array(comp, dtype=np.int32))
    return clusters

def body_to_virtual_cam(pb: np.ndarray):
    """
    Map BODY (x forward, y left, z up) -> VirtualCam (x right, y down, z forward)
    """
    if pb.size == 0:
        return pb
    X_cam = -pb[:,1]    # right
    Y_cam = -pb[:,2]    # down
    Z_cam =  pb[:,0]    # forward
    pc = np.stack([X_cam, Y_cam, Z_cam], axis=1)
    return pc

def project_pc(pc: np.ndarray):
    """
    Pinhole projection with FX/FY/CX/CY. Only keep Z>0.
    Returns (uv, depth, mask).
    """
    if pc.size == 0:
        return np.zeros((0,2), np.int32), np.zeros((0,), np.float32), np.zeros((0,), bool)
    Z = pc[:,2]
    mask = Z > 0.2
    pcv = pc[mask]
    u = (FX * (pcv[:,0] / pcv[:,2]) + CX).astype(np.int32)
    v = (FY * (pcv[:,1] / pcv[:,2]) + CY).astype(np.int32)
    uv = np.stack([u,v], axis=1)
    depth = pcv[:,2].astype(np.float32)
    return uv, depth, mask

def draw_frame(pb: np.ndarray, clusters: List[np.ndarray], use_palette=True):
    """
    Render a frame with projected points colored by cluster.
    """
    img = np.full((WIN_H, WIN_W, 3), 20, np.uint8)

    # Horizon line (just for reference)
    cv2.line(img, (0, int(CY)), (WIN_W-1, int(CY)), (60,60,60), 1, cv2.LINE_AA)

    pc = body_to_virtual_cam(pb)
    uv, depth, mask = project_pc(pc)

    # Safety bounds
    if uv.shape[0] == 0:
        cv2.putText(img, "No LiDAR points", (16,36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200), 2, cv2.LINE_AA)
        return img

    # Build a per-point color buffer
    colors = np.full((uv.shape[0],3), (180,180,180), np.uint8)  # default gray

    # Color by cluster
    if clusters:
        # cluster indices refer to pb's indexing BEFORE mask
        # create map from original index -> compact masked index
        idx_map = -np.ones(pb.shape[0], dtype=np.int32)
        idx_map[np.where(mask)[0]] = np.arange(uv.shape[0], dtype=np.int32)

        for ci, inds in enumerate(clusters):
            color = PALETTE[ci % len(PALETTE)] if use_palette else (0,255,0)
            mapped = idx_map[inds]
            mapped = mapped[mapped >= 0]
            if mapped.size:
                colors[mapped] = np.array(color, np.uint8)

    # Depth cue: smaller points when far, clamp sizes between 1..4 px
    z = depth
    size = np.clip((8.0 / (z / 5.0 + 1e-3)), 1.0, 4.0).astype(np.int32)

    # Draw points
    for (u,v), s, c in zip(uv, size, colors):
        if 0 <= u < WIN_W and 0 <= v < WIN_H:
            cv2.circle(img, (int(u),int(v)), int(s), tuple(int(x) for x in c), -1)

    # Heads-up text
    info = f"points: {pb.shape[0]}  clusters: {len(clusters)}  eps[m]: {EPS:.2f}  min_pts: {MIN_PTS}  ground:{'on' if GROUND else 'off'}"
    cv2.putText(img, info, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230,230,230), 1, cv2.LINE_AA)
    cv2.putText(img, "+X fwd  (depth),  +Y left  (→ screen left),  +Z up  (→ screen up)", (12, WIN_H-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1, cv2.LINE_AA)

    return img

def main():
    global EPS, MIN_PTS, GROUND
    EPS, MIN_PTS = EPS_INIT, MIN_PTS_INIT
    GROUND = True

    client = fsds.FSDSClient()
    client.enableApiControl(False)

    print("Keys: q quit | [ / ] eps | - / = min_pts | g ground filter | r reset")
    while True:
        pb, ts = get_lidar_points(client)
        pb = filter_body_points(pb, use_ground_filter=GROUND)

        # cluster in XY
        clusters = cluster_xy(pb, EPS, MIN_PTS)

        # draw
        frame = draw_frame(pb, clusters, use_palette=True)
        cv2.imshow("LiDAR 3D (car perspective)", frame)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'): break
        elif k == ord('['): EPS = max(0.05, EPS - 0.02)
        elif k == ord(']'): EPS = min(1.00, EPS + 0.02)
        elif k == ord('-'): MIN_PTS = max(3, MIN_PTS - 1)
        elif k == ord('='): MIN_PTS = min(40, MIN_PTS + 1)
        elif k == ord('g'): GROUND = not GROUND
        elif k == ord('r'):
            EPS, MIN_PTS, GROUND = EPS_INIT, MIN_PTS_INIT, True

    cv2.destroyAllWindows()
    client.enableApiControl(False)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
