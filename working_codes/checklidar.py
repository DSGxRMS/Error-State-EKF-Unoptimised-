# checklidar.py
# Minimal, robust LiDAR top-down viewer using OpenCV (no Matplotlib).
# - Connects to FSDS via fsds.FSDSClient
# - Reads vehicle pose and LiDAR point cloud
# - Transforms LiDAR points from the vehicle body frame into the world frame
# - Renders a 2D orthographic top-down map with axes grid, LiDAR points,
#   vehicle footprint and travelled path history.
#
# Design / theory notes (short):
#  - Sensor frames: LiDAR provides points in the vehicle body frame. We convert
#    body -> world using the vehicle pose (rotation + translation). The pose's
#    quaternion is converted to a rotation matrix. Normalising the quaternion
#    before conversion protects against small numerical errors.
#  - ENU vs NED: Some simulators / sensors use North-East-Down (NED) body axes.
#    We include a boolean to flip Y/Z if the body frame is NED to work in a
#    common ENU (East-North-Up) convention for rendering. This avoids swapped
#    axes or inverted maps.
#  - Rendering: We map a square world window (meters) centered on the vehicle
#    to canvas pixels. Points outside the window are clipped early for speed.
#  - Performance: We limit rendered LiDAR points per frame (LIDAR_MAX_POINTS).
#    Drawing points by vectorized indexing into the image array is much faster
#    than looping per-point in Python.
#
# Usage: python checklidar.py
# Keys: q = quit, p = pause, r = reset path

import time
import math
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import cv2

# FSDS client
import fsds  # ensure the FSDS python client package is available

# ----------------------------
# Configuration (tweak here)
# ----------------------------
VEHICLE_NAME = "FSCar"
LIDAR_NAME = "Lidar"

# World window half-size (meters). Rendered area will be a square of width 2*WORLD_HALF_WIDTH_M.
WORLD_HALF_WIDTH_M = 60.0

# Canvas size (pixels)
CANVAS_W, CANVAS_H = 900, 900

# Throttle: max number of lidar points drawn per frame.
LIDAR_MAX_POINTS = 80000

# If FSDS body LiDAR frame is NED, set this False to convert to ENU for rendering.
ASSUME_LIDAR_BODY_IS_ENU = True

# Car footprint (meters) for simple visualization
CAR_LENGTH_M, CAR_WIDTH_M = 1.8, 0.9

# Grid spacing in meters (visual reference)
GRID_SPACING_M = 10.0

# Maximum stored path history points
MAX_PATH_HISTORY = 5000

# ----------------------------
# Pose & transforms
# ----------------------------
@dataclass
class Pose:
    p: np.ndarray    # shape (3,) world position [x, y, z]
    R: np.ndarray    # shape (3,3) rotation matrix world_from_body


def quat_to_rotmatrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """
    Convert quaternion (x, y, z, w) to 3x3 rotation matrix.
    We normalise the quaternion to protect against numerical drift from the source.
    Rotation maps vectors from body-frame into world-frame when used as: R @ v_body.
    """
    q = np.array([qx, qy, qz, qw], dtype=float)
    n = np.linalg.norm(q)
    if n < 1e-12:
        # Degenerate quaternion; return identity (avoid NaNs)
        return np.eye(3, dtype=float)
    q /= n
    x, y, z, w = q
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    R = np.array([
        [1 - 2 * (yy + zz),     2 * (xy - wz),         2 * (xz + wy)],
        [    2 * (xy + wz), 1 - 2 * (xx + zz),         2 * (yz - wx)],
        [    2 * (xz - wy),     2 * (yz + wx),     1 - 2 * (xx + yy)]
    ], dtype=float)
    return R


def get_vehicle_pose(client: fsds.FSDSClient, vehicle_name: str = VEHICLE_NAME) -> Pose:
    """
    Query the FSDS client for the vehicle's kinematic state and produce a Pose.
    The returned rotation R maps body-frame vectors into world-frame (world_from_body).
    """
    st = client.getCarState(vehicle_name=vehicle_name)
    pos = st.kinematics_estimated.position
    ori = st.kinematics_estimated.orientation
    p = np.array([pos.x_val, pos.y_val, pos.z_val], dtype=float)
    Rwb = quat_to_rotmatrix(ori.x_val, ori.y_val, ori.z_val, ori.w_val)
    return Pose(p=p, R=Rwb)


def body_to_world(points_body: np.ndarray, pose: Pose) -> np.ndarray:
    """
    Transform Nx3 points from body frame into world frame:
      p_world = R_world_from_body @ p_body + p_world_origin
    Keep empty-array handling defensive.
    """
    if points_body.size == 0:
        return points_body.reshape(0, 3)
    return (pose.R @ points_body.T).T + pose.p


def get_lidar_points(client: fsds.FSDSClient, lidar_name: str = LIDAR_NAME,
                     vehicle_name: str = VEHICLE_NAME) -> Tuple[np.ndarray, float]:
    """
    Fetch LiDAR point cloud from FSDS client.
    Returns:
      - pts: array shape (N,3) dtype float64; empty array if no points
      - timestamp: float seconds (best-effort)
    Notes:
      - FSDS point_cloud is typically a flat array [x0,y0,z0, x1,y1,z1, ...].
      - We perform basic validation and support timestamps in ns or s.
      - If the sensor's body frame is NED, flip Y and Z to convert to ENU.
    """
    data = client.getLidarData(lidar_name=lidar_name, vehicle_name=vehicle_name)
    if data is None or data.point_cloud is None:
        return np.empty((0, 3), dtype=float), 0.0

    arr = np.asarray(data.point_cloud, dtype=np.float32)
    if arr.size == 0 or (arr.size % 3) != 0:
        # Invalid/empty point cloud
        t = float(data.time_stamp) * 1e-9 if data.time_stamp > 1e12 else float(data.time_stamp)
        return np.empty((0, 3), dtype=float), t

    pts = arr.reshape(-1, 3).astype(np.float64)

    # Convert body axes if necessary (NED -> ENU)
    if not ASSUME_LIDAR_BODY_IS_ENU:
        # NED: x->east, y->north, z->down; to get ENU we flip Y and Z
        pts[:, 1] *= -1.0
        pts[:, 2] *= -1.0

    t = float(data.time_stamp) * 1e-9 if data.time_stamp > 1e12 else float(data.time_stamp)
    return pts, t


# ----------------------------
# Rendering helpers (OpenCV)
# ----------------------------
def world_to_canvas(xy_world: np.ndarray, center_xy: np.ndarray,
                    hw_m: float, w: int, h: int) -> np.ndarray:
    """
    Map world XY (meters) in a square window around center_xy into pixel coords.
    - xy_world: (N,2) array of (x,y) world coordinates in meters
    - center_xy: (2,) center of the square window in world coords (vehicle position)
    - hw_m: half-width of window in meters
    - returns: (M,2) int32 pixel coords for points that fall inside the window
    Coordinate conventions:
      - world x increases to the right (east), world y increases upward (north)
      - image x (cols) increase to right, image y (rows) increase downward: so we flip Y
    """
    if xy_world.size == 0:
        return np.empty((0, 2), dtype=np.int32)

    dx = (xy_world[:, 0] - center_xy[0]) / hw_m
    dy = (xy_world[:, 1] - center_xy[1]) / hw_m

    mask = (np.abs(dx) <= 1.0) & (np.abs(dy) <= 1.0)
    if not np.any(mask):
        return np.empty((0, 2), dtype=np.int32)

    dx = dx[mask]
    dy = dy[mask]

    px = ((dx + 1.0) * 0.5 * (w - 1)).astype(np.int32)
    py = ((1.0 - (dy + 1.0) * 0.5) * (h - 1)).astype(np.int32)
    return np.stack([px, py], axis=1)


def render_world_to_bgr(canvas_w: int, canvas_h: int,
                        pose: Pose,
                        pts_world: np.ndarray,
                        path_xy: List[List[float]],
                        hw_m: float = WORLD_HALF_WIDTH_M) -> np.ndarray:
    """
    Build a BGR image for the top-down map.
    Layers (top -> bottom):
      - white background
      - light grid lines (meters)
      - LiDAR points (brownish)
      - path polyline (black)
      - vehicle footprint (red) + centre marker (cross)
      - title strip
    Performance: vectorized drawing of LiDAR points by indexing into image array.
    """
    img = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)  # white

    # ---------- grid ----------
    for m in np.arange(-hw_m, hw_m + 1e-6, GRID_SPACING_M):
        # vertical line at x = pose.p[0] + m
        p1 = world_to_canvas(np.array([[pose.p[0] + m, pose.p[1]]]), pose.p[:2], hw_m, canvas_w, canvas_h)
        p2 = world_to_canvas(np.array([[pose.p[0] + m, pose.p[1] + 2 * hw_m]]), pose.p[:2], hw_m, canvas_w, canvas_h)
        if p1.size and p2.size:
            cv2.line(img, tuple(p1[0]), tuple(p2[0]), (230, 230, 230), 1)

        # horizontal line at y = pose.p[1] + m
        p3 = world_to_canvas(np.array([[pose.p[0], pose.p[1] + m]]), pose.p[:2], hw_m, canvas_w, canvas_h)
        p4 = world_to_canvas(np.array([[pose.p[0] + 2 * hw_m, pose.p[1] + m]]), pose.p[:2], hw_m, canvas_w, canvas_h)
        if p3.size and p4.size:
            cv2.line(img, tuple(p3[0]), tuple(p4[0]), (230, 230, 230), 1)

    # ---------- LiDAR points ----------
    if pts_world.shape[0] > 0:
        xy = pts_world[:, :2]

        # Throttle points to cap CPU/GPU work per frame
        n = xy.shape[0]
        if n > LIDAR_MAX_POINTS:
            step = int(math.ceil(n / float(LIDAR_MAX_POINTS)))
            xy = xy[::step]

        pts_px = world_to_canvas(xy, pose.p[:2], hw_m, canvas_w, canvas_h)
        if pts_px.shape[0] > 0:
            # vectorized pixel set: guard for in-bounds (should hold after world_to_canvas clipping)
            xs = pts_px[:, 0]
            ys = pts_px[:, 1]
            valid = (xs >= 0) & (xs < canvas_w) & (ys >= 0) & (ys < canvas_h)
            xs = xs[valid]; ys = ys[valid]
            # set colour by advanced indexing (BGR)
            img[ys, xs] = (200, 100, 50)

    # ---------- Path history ----------
    if len(path_xy) >= 2:
        path_np = np.array(path_xy, dtype=float)
        path_px = world_to_canvas(path_np, pose.p[:2], hw_m, canvas_w, canvas_h)
        if path_px.shape[0] >= 2:
            cv2.polylines(img, [path_px], False, (0, 0, 0), 1, lineType=cv2.LINE_AA)

    # ---------- Vehicle footprint (triangle) ----------
    tri_body = np.array([
        [+CAR_LENGTH_M / 2, 0.0, 0.0],
        [-CAR_LENGTH_M / 2, +CAR_WIDTH_M / 2, 0.0],
        [-CAR_LENGTH_M / 2, -CAR_WIDTH_M / 2, 0.0]
    ], dtype=float)
    tri_world = (pose.R @ tri_body.T).T + pose.p
    tri_px = world_to_canvas(tri_world[:, :2], pose.p[:2], hw_m, canvas_w, canvas_h)
    if tri_px.shape[0] == 3:
        cv2.polylines(img, [tri_px], True, (0, 0, 255), 2, lineType=cv2.LINE_AA)

    # Crosshair at vehicle position
    car_px = world_to_canvas(np.array([pose.p[:2]]), pose.p[:2], hw_m, canvas_w, canvas_h)
    if car_px.size:
        cx, cy = int(car_px[0, 0]), int(car_px[0, 1])
        cv2.drawMarker(img, (cx, cy), (0, 0, 0), markerType=cv2.MARKER_TILTED_CROSS, markerSize=10, thickness=1)

    # Title strip
    cv2.rectangle(img, (0, 0), (canvas_w, 24), (245, 245, 245), -1)
    title = f"LiDAR Top-Down (world)  W=±{hw_m:.0f} m"
    cv2.putText(img, title, (10, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 10, 10), 1, cv2.LINE_AA)

    return img


# ----------------------------
# Main application loop
# ----------------------------
def main() -> None:
    client = fsds.FSDSClient()
    # Make sure control flag is off (we only read state)
    client.enableApiControl(False)

    path_hist: List[List[float]] = []
    paused = False

    print("[INFO] LiDAR viewer running. Keys: q=quit, p=pause, r=reset path")

    try:
        while True:
            if not paused:
                # Read vehicle pose (defensive: wrap FSDS queries so a client error doesn't crash us)
                try:
                    pose = get_vehicle_pose(client, VEHICLE_NAME)
                except Exception as e:
                    # If state couldn't be read, print and skip this frame
                    print(f"[WARN] Could not read vehicle pose: {e}")
                    time.sleep(0.05)
                    continue

                # Read lidar, transform to world frame
                try:
                    lidar_body, _ = get_lidar_points(client, LIDAR_NAME, VEHICLE_NAME)
                except Exception as e:
                    print(f"[WARN] Could not read LiDAR points: {e}")
                    lidar_body = np.empty((0, 3), dtype=float)

                lidar_world = body_to_world(lidar_body, pose)

                # Append path history (store X,Y only)
                path_hist.append([float(pose.p[0]), float(pose.p[1])])
                if len(path_hist) > MAX_PATH_HISTORY:
                    path_hist = path_hist[-MAX_PATH_HISTORY:]

                # Render image and show
                canvas = render_world_to_bgr(CANVAS_W, CANVAS_H, pose, lidar_world, path_hist, WORLD_HALF_WIDTH_M)
                cv2.imshow("LiDAR Top-Down", canvas)

            # Key handling AFTER imshow so window repaints
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = not paused
            elif key == ord('r'):
                path_hist = []

            # When paused, reduce CPU usage
            if paused:
                time.sleep(0.05)

    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        client.enableApiControl(False)


if __name__ == "__main__":
    main()
