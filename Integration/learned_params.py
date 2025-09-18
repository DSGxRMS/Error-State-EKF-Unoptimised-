# learned_params.py
from __future__ import annotations
import json, os
from dataclasses import dataclass, asdict

@dataclass
class LearnedParams:
    Rv_baseline: float = 0.040   # speed meas noise
    gyro_bias: float   = 0.0     # rad/s, yaw-rate bias to subtract from IMU
    v_scale: float     = 1.0     # multiplicative scale for wheel speed

    @staticmethod
    def load(path: str) -> "LearnedParams":
        try:
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
            return LearnedParams(
                Rv_baseline=float(d.get("Rv_baseline", 0.040)),
                gyro_bias=float(d.get("gyro_bias", 0.0)),
                v_scale=float(d.get("v_scale", 1.0)),
            )
        except Exception:
            return LearnedParams()

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    def smooth_update(self, Rv=None, gyro_bias=None, v_scale=None, alpha=0.2):
        # exponential moving average toward new values
        if Rv is not None:
            self.Rv_baseline = float(max(1e-4, (1-alpha)*self.Rv_baseline + alpha*float(Rv)))
        if gyro_bias is not None:
            self.gyro_bias   = float((1-alpha)*self.gyro_bias + alpha*float(gyro_bias))
        if v_scale is not None:
            self.v_scale     = float((1-alpha)*self.v_scale + alpha*float(v_scale))
