# Live stereo camera visualizer for FSDS using fsds.FSDSClient
# - Streams camera "Cam_L" and "Cam_R"
# - Applies saved tuning (contrast/brightness/saturation/gamma) BEFORE YOLO
# - Runs YOLOv5 (RGB) and overlays detections
# - Displays the tuned frames (so display == YOLO input)
#
# Deps: numpy, opencv-python, torch
# YOLOv5 repo expected at ./yolov5 with weights at ./yolov5/weights/best.pt

import os
import sys
import time
import math
import json
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import cv2
import torch


sys.path.insert(0, os.path.abspath("./yolov5"))
from yolov5.utils.augmentations import letterbox
from yolov5.utils.general import non_max_suppression, scale_boxes

# --- FSDS client ---
import fsds


# ===================== CONFIG =====================
VEHICLE_NAME = "FSCar"
CAM_LEFT     = "Cam_L"   
CAM_RIGHT    = "Cam_R"

YOLO_WEIGHTS = "./yolov5/weights/best.pt"
YOLO_SIZE    = 640
YOLO_CONF    = 0.75
YOLO_IOU     = 0.35

# Path to the tuning JSON you saved with the tuner tool
TUNING_JSON  = "./camera_tuning/tuning_latest.json"  
# ==================================================


# ===================== TUNING FILTER =====================
@dataclass
class TuningParams:
    contrast_x100: int = 100   # alpha = contrast_x100 / 100
    brightness_off: int = 100  # beta  = brightness_off - 100  (range -100..+100)
    saturation_x100: int = 100 # sat_mult = saturation_x100 / 100
    gamma_x100: int = 100      # gamma = gamma_x100 / 100

    @property
    def alpha(self) -> float:
        return max(0.0, self.contrast_x100 / 100.0)

    @property
    def beta(self) -> float:
        return float(self.brightness_off - 100)

    @property
    def sat_mult(self) -> float:
        return max(0.0, self.saturation_x100 / 100.0)

    @property
    def gamma(self) -> float:
        return max(0.10, self.gamma_x100 / 100.0)


class TuningFilter:
    """Applies contrast/brightness → saturation → gamma. Accepts/returns RGB."""
    def __init__(self, params: Optional[TuningParams] = None):
        self.params = params or TuningParams()
        self._gamma_val: Optional[float] = None
        self._gamma_lut: Optional[np.ndarray] = None

    @staticmethod
    def _build_gamma_lut(gamma: float) -> np.ndarray:
        inv = 1.0 / gamma
        x = np.arange(256, dtype=np.float32) / 255.0
        lut = np.clip(np.power(x, inv) * 255.0, 0, 255).astype(np.uint8)
        return lut

    def _ensure_gamma(self):
        if self._gamma_val != self.params.gamma or self._gamma_lut is None:
            self._gamma_lut = self._build_gamma_lut(self.params.gamma)
            self._gamma_val = self.params.gamma

    def apply_rgb(self, rgb: np.ndarray) -> np.ndarray:
        """Return tuned RGB image (uint8)."""
        if rgb is None:
            return None
        # Work in BGR for OpenCV color ops, convert back to RGB at end
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # 1) Contrast/Brightness
        bgr = cv2.convertScaleAbs(bgr, alpha=self.params.alpha, beta=self.params.beta)

        # 2) Saturation via HSV S scaling
        if self.params.saturation_x100 != 100:
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            s = np.clip((self.params.sat_mult * s).astype(np.float32), 0, 255).astype(np.uint8)
            hsv = cv2.merge([h, s, v])
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # 3) Gamma via LUT
        self._ensure_gamma()
        bgr = cv2.LUT(bgr, self._gamma_lut)

        # Back to RGB for the rest of your pipeline
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_tuning_params(json_path: str) -> TuningParams:
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        params_dict = data.get("params", data)
        return TuningParams(
            contrast_x100=int(params_dict.get("contrast_x100", 100)),
            brightness_off=int(params_dict.get("brightness_off", 100)),
            saturation_x100=int(params_dict.get("saturation_x100", 100)),
            gamma_x100=int(params_dict.get("gamma_x100", 100)),
        )
    except Exception as e:
        print(f"[WARN] Could not load tuning JSON '{json_path}': {e}\n       Using identity tuning.")
        return TuningParams()
# =========================================================


# ===================== YOLO =======================
class YOLOv5Detector:
    def __init__(self, repo_dir: str = "./yolov5", weights_path: str = YOLO_WEIGHTS, device: str = "cuda"):
        import pathlib
        pathlib.PosixPath = pathlib.WindowsPath  # allow Linux-saved checkpoints on Windows

        sys.path.insert(0, repo_dir)
        from models.common import DetectMultiBackend

        self.device = torch.device(device)
        self.model = DetectMultiBackend(weights_path, device=self.device, dnn=False, data=None, fp16=False)
        self.names = self.model.names

        # Colors in RGB (we keep frames in RGB until show-time)
        self.cls_color_rgb = {
            "yellow_cone":       (255, 255,   0),
            "blue_cone":         (  0,   0, 255),
            "orange_cone":       (255, 165,   0),
            "large_orange_cone": (255, 255, 255),  
            "unknown_cone":      (  0, 255,   0)
        }

    def infer(self, img_rgb: np.ndarray):
        """Run YOLOv5 on an RGB image and return detections."""
        im = letterbox(img_rgb, new_shape=YOLO_SIZE, stride=32, auto=True)[0]
        im = im.transpose((2, 0, 1))  # HWC->CHW
        im = np.ascontiguousarray(im)
        im = torch.from_numpy(im).to(self.device).float() / 255.0
        if im.ndim == 3:
            im = im.unsqueeze(0)

        pred = self.model(im)
        pred = non_max_suppression(pred, conf_thres=YOLO_CONF, iou_thres=YOLO_IOU)[0]

        out = []
        if pred is None or len(pred) == 0:
            return out

        pred[:, :4] = scale_boxes(im.shape[2:], pred[:, :4], img_rgb.shape).round()
        for *xyxy, conf, cls in pred.cpu().tolist():
            cls = int(cls)
            name = self.names[cls] if cls in self.names else str(cls)
            x1, y1, x2, y2 = map(int, xyxy)
            out.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "conf": float(conf), "cls": cls, "name": name})
        return out

    def draw_rgb(self, img_rgb: np.ndarray, dets):
        """Draw boxes directly on an RGB image (so colors below are RGB tuples)."""
        if not img_rgb.flags.writeable:
            img_rgb = img_rgb.copy()
        for d in dets:
            x1, y1, x2, y2 = d["x1"], d["y1"], d["x2"], d["y2"]
            name, conf = d["name"], d["conf"]
            color = self.cls_color_rgb.get(name, (0, 255, 0))  # RGB color on RGB image
            cv2.rectangle(img_rgb, (x1, y1), (x2, y2), color, 2)
            label = f"{name} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            # label box uses same color; text is black for contrast even on white
            cv2.rectangle(img_rgb, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(img_rgb, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        return img_rgb
# ==================================================


# ===================== CAMERA IO ==================
def get_camera_image(client, cam_name: str, vehicle_name: str = VEHICLE_NAME):
    """
    Return RGB uint8 image (writeable).
    FSDS may return BGRA; convert to RGB once and keep pipeline in RGB.
    """
    req = [fsds.ImageRequest(cam_name, fsds.ImageType.Scene, pixels_as_float=False, compress=False)]
    rsp = client.simGetImages(req, vehicle_name=vehicle_name)
    if not rsp or rsp[0].width == 0:
        return None

    h, w = rsp[0].height, rsp[0].width
    buf = np.frombuffer(rsp[0].image_data_uint8, dtype=np.uint8).copy()  # writeable

    if buf.size == 4 * h * w:
        img = buf.reshape(h, w, 4)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    elif buf.size == 3 * h * w:
        img_bgr = buf.reshape(h, w, 3)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = buf.reshape(h, w, -1)

    return img_rgb.copy()
# ==================================================


# ===================== MAIN =======================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    yolo = YOLOv5Detector(weights_path=YOLO_WEIGHTS, device=device)

    # Load tuning once
    tuning_params = load_tuning_params(TUNING_JSON)
    tuner = TuningFilter(tuning_params)
    print(f"[TUNING] alpha={tuning_params.alpha:.2f}  beta={int(tuning_params.beta):+d}  "
          f"sat={tuning_params.sat_mult:.2f}  gamma={tuning_params.gamma:.2f}")

    client = fsds.FSDSClient()
    client.enableApiControl(False)

 
    img0 = get_camera_image(client, CAM_LEFT)
    if img0 is not None:
        h, w = img0.shape[:2]
        nominal_fov_deg = 90.0
        fx = (w/2) / math.tan(math.radians(nominal_fov_deg/2))
        # (silent)

    print("Press 'q' to quit, 'p' to pause.")
    paused = False

    while True:
        if not paused:
            # 1) Acquire raw RGB
            img_l = get_camera_image(client, CAM_LEFT)
            img_r = get_camera_image(client, CAM_RIGHT)

            # 2) Apply tuning BEFORE YOLO (and we also DISPLAY these tuned frames)
            if img_l is not None:
                tuned_l = tuner.apply_rgb(img_l)        # tuned for both YOLO + display
                dets_l  = yolo.infer(tuned_l)
                tuned_l = yolo.draw_rgb(tuned_l, dets_l)
                cv2.imshow("Camera - Left (tuned + YOLO)", cv2.cvtColor(tuned_l, cv2.COLOR_RGB2BGR))

            if img_r is not None:
                tuned_r = tuner.apply_rgb(img_r)
                dets_r  = yolo.infer(tuned_r)
                tuned_r = yolo.draw_rgb(tuned_r, dets_r)
                cv2.imshow("Camera - Right (tuned + YOLO)", cv2.cvtColor(tuned_r, cv2.COLOR_RGB2BGR))

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('p'):
            paused = not paused
            time.sleep(0.05)

    cv2.destroyAllWindows()
    client.enableApiControl(False)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
