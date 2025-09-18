"""
sensors_fsds.py
FSDS sensor readers for EKF spine:
- IMU: wz (rad/s), ax (m/s^2), timestamp [s]
- Speed: CarState.speed (m/s), timestamp [s]

Assumes settings.json includes an IMU named "Imu".
"""

from __future__ import annotations
from dataclasses import dataclass
import time
import fsds

@dataclass
class FSDSConfig:
    vehicle_name: str = "FSCar"
    imu_name: str = "Imu"

class FSDSReaders:
    def __init__(self, cfg: FSDSConfig = FSDSConfig()):
        self.cfg = cfg
        self.client = fsds.FSDSClient()
        self.client.confirmConnection()
        # do not enable API control here; the controller may manage it

    def imu(self):
        """
        Returns (wz, ax, tsec)
        wz: yaw rate [rad/s] (up-axis)
        ax: longitudinal accel [m/s^2] (vehicle x-forward)
        tsec: timestamp [s] (converted from ns in sim)
        """
        imu = self.client.getImuData(imu_name=self.cfg.imu_name, vehicle_name=self.cfg.vehicle_name)
        wz = float(getattr(imu.angular_velocity, "z_val", 0.0))
        ax = float(getattr(imu.linear_acceleration, "x_val", 0.0))
        tsec = float(getattr(imu, "time_stamp", 0)) * 1e-9
        # Fallback if stamp missing
        if tsec <= 0.0:
            tsec = time.time()
        return wz, ax, tsec

    def speed(self):
        """
        Returns (v, tsec), using CarState.speed as wheel-speed surrogate.
        """
        st = self.client.getCarState()
        v = float(getattr(st, "speed", 0.0))
        return v, time.time()

    def release(self):
        try:
            self.client.enableApiControl(False, self.cfg.vehicle_name)
        except Exception:
            pass
