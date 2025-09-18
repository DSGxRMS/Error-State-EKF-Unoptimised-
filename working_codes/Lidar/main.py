#!/usr/bin/env python3
import sys
import os
import time
import fsds
import numpy as np
import python.lidar_ekf.visuals.Lidar.ransacc as ransacc
import pandas as pd
import lidar_pipeline
# Qt + PyQtGraph GL
from PySide6 import QtCore, QtWidgets
import pyqtgraph as pg
import pyqtgraph.opengl as gl

# FSDS SDK
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --------- math helpers ---------
def quat_to_R3(qw, qx, qy, qz):
    """Return 3x3 rotation matrix from quaternion (w, x, y, z)."""
    # normalized (FSDS generally gives unit quats)
    n = np.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    if n == 0:  # fall back
        return np.eye(3, dtype=np.float32)
    qw, qx, qy, qz = qw/n, qx/n, qy/n, qz/n
    xx, yy, zz = qx*qx, qy*qy, qz*qz
    xy, xz, yz = qx*qy, qx*qz, qy*qz
    wx, wy, wz = qw*qx, qw*qy, qw*qz
    R = np.array([
        [1 - 2*(yy+zz),     2*(xy - wz),       2*(xz + wy)],
        [    2*(xy + wz), 1 - 2*(xx+zz),       2*(yz - wx)],
        [    2*(xz - wy),     2*(yz + wx),   1 - 2*(xx+yy)]
    ], dtype=np.float32)
    return R


def fsds_lidar_world_points(lidar):
    """Convert FSDS lidar frame to Nx3 world points using the lidar pose at capture."""
    pts = np.asarray(lidar.point_cloud, dtype=np.float32).reshape(-1, 3)
    # Pose is in global frame; rotation maps sensor->world
    pos = lidar.pose.position
    ori = lidar.pose.orientation
    p = np.array([pos.x_val, pos.y_val, pos.z_val], dtype=np.float32)
    R = quat_to_R3(ori.w_val, ori.x_val, ori.y_val, ori.z_val)
    # world = R @ sensor + p
    if pts.size == 0:
        return pts
    return (pts @ R.T) + p


# --------- main viewer ---------
class LidarViewer(QtWidgets.QMainWindow):
    def __init__(self, lidar_name="Lidar", decimate=2, point_size=2.0):
        super().__init__()
        self.setWindowTitle("FSDS LiDAR Viewer (PyQtGraph)")
        self.resize(1000, 800)

        self.client = fsds.FSDSClient()
        self.client.confirmConnection()

        self.lidar_name = lidar_name
        self.decimate = max(1, int(decimate))
        self.point_size = point_size

        # GL view widget
        self.view = gl.GLViewWidget()
        self.view.setCameraPosition(distance=30)  # pull back a bit
        self.setCentralWidget(self.view)

        # Ground grid
        g = gl.GLGridItem()
        g.scale(2, 2, 1)
        g.setSize(100, 100)
        self.view.addItem(g)

        # Point cloud item
        self.pcd = gl.GLScatterPlotItem()
        self.pcd.setGLOptions('opaque')  # good perf
        self.pcd.setData(pos=np.zeros((1,3), dtype=np.float32), size=self.point_size)
        self.view.addItem(self.pcd)

        # Timer for realtime updates
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # ~33 FPS; adjust if needed

        # Simple FPS meter
        self._last = time.time()
        self._frames = 0

    @QtCore.Slot()
    def update_frame(self):
        lidar = self.client.getLidarData(lidar_name=self.lidar_name)
        vel_msg = fsds_lidar_world_points(lidar)
        cone_locations = []
        final_points = []
        if vel_msg.size:
            # Decimate to keep FPS high if needed
            if self.decimate > 1:
                vel_msg = vel_msg[::self.decimate]

            point_cloud = pd.DataFrame(vel_msg, columns=["X" , "Y" ,"Z"])
            algo = ransacc.RANSAC(point_cloud, max_iterations=25, distance_ratio_threshold=0.01)
            ground_points, non_ground_points = algo._ransac_algorithm()

            non_ground_points = non_ground_points.to_numpy()
            clusters = lidar_pipeline.euclidean_clustering(non_ground_points, 0.5)

            for cluster in clusters:
                cluster = non_ground_points[cluster]
                final_points.extend(cluster)
                mean = np.mean(cluster, axis=0)
                cone_locations.append(mean)

            self.pcd.setData(pos=final_points, size=self.point_size)
            print(len(cone_locations))   # cone_locations file contains the positions of cones

        # crude FPS
        self._frames += 1
        now = time.time()
        if now - self._last >= 1.0:
            fps = self._frames/(now - self._last)
            self.setWindowTitle(f"FSDS LiDAR Viewer (PyQtGraph) — {fps:.1f} FPS")
            self._frames = 0
            self._last = now


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = LidarViewer(lidar_name="Lidar", decimate=2, point_size=2.0)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
