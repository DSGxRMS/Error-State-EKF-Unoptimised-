# Live stereo camera visualizer for FSDS using fsds.FSDSClient
# - Streams camera "Cam_L" and "Cam_R"
# - Runs YOLOv5 (RGB) and overlays detections
# - Displays with OpenCV (BGR)
#
# Deps: numpy, opencv-python, torch
# YOLOv5 repo expected at ./yolov5 with weights at ./yolov5/weights/best.pt

import os
import sys
import time
import math
from dataclasses import dataclass  # kept if you later add pose utilities
from typing import Tuple
import numpy as np
import cv2
import torch

# --- Ensure yolov5 repo on path (so utils.* resolves) ---
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
YOLO_IOU     = 0.45
# ==================================================


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
            "yellow_cone":      (255, 255,   0),
            "blue_cone":        (  0,   0, 255),
            "orange_cone":      (255, 165,   0),
            "large_orange_cone":(255, 140,   0),
            "unknown_cone":     (  0, 255,   0)
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
        """Draw boxes directly on an RGB image."""
        if not img_rgb.flags.writeable:
            img_rgb = img_rgb.copy()
        for d in dets:
            x1, y1, x2, y2 = d["x1"], d["y1"], d["x2"], d["y2"]
            name, conf = d["name"], d["conf"]
            color = self.cls_color_rgb.get(name, (0, 255, 0))
            cv2.rectangle(img_rgb, (x1, y1), (x2, y2), color, 2)
            label = f"{name} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
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

    client = fsds.FSDSClient()
    client.enableApiControl(False)

    # Approx intrinsics from FOV to help future overlays if needed
    img0 = get_camera_image(client, CAM_LEFT)
    if img0 is not None:
        h, w = img0.shape[:2]
        nominal_fov_deg = 90.0
        fx = (w/2) / math.tan(math.radians(nominal_fov_deg/2))
      

    print("Press 'q' to quit, 'p' to pause.")
    paused = False

    while True:
        if not paused:
            img_l = get_camera_image(client, CAM_LEFT)
            img_r = get_camera_image(client, CAM_RIGHT)

            if img_l is not None:
                dets_l = yolo.infer(img_l)
                img_l = yolo.draw_rgb(img_l, dets_l)
                cv2.imshow("Camera - Left (YOLO)", cv2.cvtColor(img_l, cv2.COLOR_RGB2BGR))

            if img_r is not None:
                dets_r = yolo.infer(img_r)
                img_r = yolo.draw_rgb(img_r, dets_r)
                cv2.imshow("Camera - Right (YOLO)", cv2.cvtColor(img_r, cv2.COLOR_RGB2BGR))

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
