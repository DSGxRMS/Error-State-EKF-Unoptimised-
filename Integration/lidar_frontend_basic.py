"""
lidar_frontend_basic.py
Minimal LiDAR frontend:
- reads local LiDAR point cloud (sensor frame) from FSDS
- ROI & height filter, voxel downsample
- simple Euclidean clustering in XY
- outputs centroids in the **body (car) frame** (not world)

No ground-truth pose is used anywhere.
"""

import numpy as np
from scipy.spatial import cKDTree
import fsds

class LidarFrontend:
    def __init__(self, vehicle_name="FSCar", lidar_name="Lidar"):
        self.client = fsds.FSDSClient()
        self.client.confirmConnection()
        self.vname = vehicle_name
        self.lname = lidar_name

    def _get_points_local(self):
        ld = self.client.getLidarData(lidar_name=self.lname, vehicle_name=self.vname)
        P = np.asarray(ld.point_cloud, dtype=np.float32).reshape(-1, 3)
        return P  # sensor/body frame

    @staticmethod
    def roi_z(P, rmin=1.0, rmax=60.0, zmin=-0.1, zmax=0.9):
        if P.size == 0: return P
        r = np.hypot(P[:,0], P[:,1])
        m = (r>=rmin)&(r<=rmax)&(P[:,2]>=zmin)&(P[:,2]<=zmax)
        return P[m]

    @staticmethod
    def voxel(P, voxel=0.08):
        if P.size == 0: return P
        keys = np.floor(P / voxel).astype(np.int64)
        _, idx = np.unique(keys, axis=0, return_index=True)
        return P[np.sort(idx)]

    @staticmethod
    def cluster_xy(P, radius=0.18, min_pts=8, max_pts=1200):
        if P.shape[0] == 0: return []
        XY = P[:, :2]
        tree = cKDTree(XY)
        n = XY.shape[0]
        visited = np.zeros(n, dtype=bool)
        clusters = []
        for i in range(n):
            if visited[i]: continue
            to_visit = [i]; visited[i]=True; comp=[]
            while to_visit:
                j = to_visit.pop()
                comp.append(j)
                nbrs = tree.query_ball_point(XY[j], r=radius)
                for k in nbrs:
                    if not visited[k]:
                        visited[k]=True; to_visit.append(k)
            if (len(comp)>=min_pts) and (len(comp)<=max_pts):
                clusters.append(np.array(comp, dtype=int))
        return clusters

    def get_centroids_body(self):
        P = self._get_points_local()
        P = self.roi_z(P, 1.0, 60.0, -0.1, 0.9)
        P = self.voxel(P, 0.08)
        clusters = self.cluster_xy(P, 0.18, 8, 1200)
        if not clusters:
            return np.empty((0,2), dtype=np.float32)
        C = np.array([P[idx].mean(axis=0) for idx in clusters], dtype=np.float32)[:, :2]
        return C  # (N,2) in body frame (x forward, y left)
