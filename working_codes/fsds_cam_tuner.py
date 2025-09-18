# Live dual-camera tuner for FSDS (NO YOLO).
# - Streams "Cam_L" and "Cam_R" from FSDS.
# - Applies contrast, brightness, saturation (and gamma) non-destructively before display.
# - Real-time controls via OpenCV trackbars.
# - 's' to save current params to JSON, 'r' to reset, 'q' to quit.
#
# Deps: numpy, opencv-python, fsds (your existing FSDS Python client)

import os
import json
import time
import math
from dataclasses import dataclass, asdict
from typing import Optional, Tuple

import numpy as np
import cv2
import fsds  # must be importable in your environment

# ===================== CONFIG =====================
VEHICLE_NAME = "FSCar"
CAM_LEFT     = "Cam_L"   # must match settings.json
CAM_RIGHT    = "Cam_R"
SAVE_DIR     = "./camera_tuning"
WINDOW_LEFT  = "Left Cam (tuned)"
WINDOW_RIGHT = "Right Cam (tuned)"
WINDOW_CTRL  = "Tuning Controls"
OSD_FONT     = cv2.FONT_HERSHEY_SIMPLEX
# ==================================================


@dataclass
class TuningParams:
    # Contrast: multiply (alpha). 1.00 = identity
    contrast_x100: int = 100   # slider 0..300 -> alpha = value/100
    # Brightness: add (beta). 0 = -100, 200 = +100
    brightness_off: int = 100  # slider 0..200 -> beta = value-100
    # Saturation multiplier in HSV. 1.00 = identity
    saturation_x100: int = 100 # slider 0..300 -> s_mult = value/100
    # Gamma correction. 1.00 = identity
    gamma_x100: int = 100      # slider 10..300 -> gamma = value/100

    def alpha(self) -> float:
        return max(0.0, self.contrast_x100 / 100.0)

    def beta(self) -> float:
        return float(self.brightness_off - 100)

    def sat_mult(self) -> float:
        return max(0.0, self.saturation_x100 / 100.0)

    def gamma(self) -> float:
        return max(0.10, self.gamma_x100 / 100.0)


def ensure_dir(d: str):
    if not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)


def get_camera_image(client, cam_name: str, vehicle_name: str = VEHICLE_NAME) -> Optional[np.ndarray]:
    """
    Return RGB uint8 image (writeable).
    FSDS may return BGRA; convert to RGB once and return.
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


def build_gamma_lut(gamma: float) -> np.ndarray:
    # gamma correction: out = 255 * (in/255)^(1/gamma)
    inv = 1.0 / gamma
    x = np.arange(256, dtype=np.float32) / 255.0
    lut = np.clip(np.power(x, inv) * 255.0, 0, 255).astype(np.uint8)
    return lut


def apply_tuning(rgb: np.ndarray, tp: TuningParams, cached) -> np.ndarray:
    """
    Apply contrast/brightness, saturation, and gamma to an RGB frame.
    Return BGR ready for imshow.
    'cached' holds reusable objects (e.g., gamma LUT) to avoid recompute.
    """
    if rgb is None:
        return None

    # Work in BGR for OpenCV ops
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # 1) Contrast/Brightness (linear): bgr' = alpha*bgr + beta
    alpha, beta = tp.alpha(), tp.beta()
    bgr = cv2.convertScaleAbs(bgr, alpha=alpha, beta=beta)

    # 2) Saturation via HSV scaling
    if tp.saturation_x100 != 100:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s = np.clip((tp.sat_mult() * s).astype(np.float32), 0, 255).astype(np.uint8)
        hsv = cv2.merge([h, s, v])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # 3) Gamma via LUT on each channel
    current_gamma = tp.gamma()
    if cached.get("gamma_val") != current_gamma or cached.get("gamma_lut") is None:
        cached["gamma_lut"] = build_gamma_lut(current_gamma)
        cached["gamma_val"] = current_gamma
    lut = cached["gamma_lut"]
    bgr = cv2.LUT(bgr, lut)

    return bgr


def put_osd(img_bgr: np.ndarray, tp: TuningParams, fps: float):
    h, w = img_bgr.shape[:2]
    text = f"C {tp.alpha():.2f}  B {tp.beta():.5f}  S {tp.sat_mult():.2f}  G {tp.gamma():.2f}  {fps:.1f} FPS"
    cv2.putText(img_bgr, text, (8, h - 12), OSD_FONT, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img_bgr, text, (8, h - 12), OSD_FONT, 0.6, (0, 0, 0),   1, cv2.LINE_AA)


def create_trackbars(tp_init: TuningParams):
    cv2.namedWindow(WINDOW_CTRL, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_CTRL, 480, 200)

    def _noop(val): pass

    cv2.createTrackbar("contrast x100 (0-300)", WINDOW_CTRL, tp_init.contrast_x100, 300, _noop)
    cv2.createTrackbar("brightness (-100..+100)", WINDOW_CTRL, tp_init.brightness_off, 200, _noop)
    cv2.createTrackbar("saturation x100 (0-300)", WINDOW_CTRL, tp_init.saturation_x100, 300, _noop)
    cv2.createTrackbar("gamma x100 (10-300)", WINDOW_CTRL, tp_init.gamma_x100, 300, _noop)
    # Set min practical gamma 10 -> 0.10
    if tp_init.gamma_x100 < 10:
        cv2.setTrackbarPos("gamma x100 (10-300)", WINDOW_CTRL, 100)


def read_trackbar_params() -> TuningParams:
    c = cv2.getTrackbarPos("contrast x100 (0-300)", WINDOW_CTRL)
    b = cv2.getTrackbarPos("brightness (-100..+100)", WINDOW_CTRL)
    s = cv2.getTrackbarPos("saturation x100 (0-300)", WINDOW_CTRL)
    g = cv2.getTrackbarPos("gamma x100 (10-300)", WINDOW_CTRL)
    g = max(10, g)
    return TuningParams(contrast_x100=c, brightness_off=b, saturation_x100=s, gamma_x100=g)


def save_params(tp: TuningParams, w_h: Tuple[int,int]):
    ensure_dir(SAVE_DIR)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(SAVE_DIR, "tuning_latest.json")
    payload = {
        "vehicle": VEHICLE_NAME,
        "cams": [CAM_LEFT, CAM_RIGHT],
        "frame_size": {"width": w_h[0], "height": w_h[1]},
        "params": asdict(tp),
        "note": "alpha = contrast_x100/100, beta = brightness_off-100, sat_mult = saturation_x100/100, gamma = gamma_x100/100"
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def main():
    client = fsds.FSDSClient()
    client.enableApiControl(False)

    # Probe size once (left cam); if unavailable, keep None
    img0 = get_camera_image(client, CAM_LEFT)
    if img0 is not None:
        h0, w0 = img0.shape[:2]
    else:
        h0, w0 = None, None

    # Windows
    cv2.namedWindow(WINDOW_LEFT,  cv2.WINDOW_NORMAL)
    cv2.namedWindow(WINDOW_RIGHT, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_LEFT,  960, 540)
    cv2.resizeWindow(WINDOW_RIGHT, 960, 540)

    tp = TuningParams()
    create_trackbars(tp)
    gamma_cache = {"gamma_lut": None, "gamma_val": None}

    print("Controls: drag sliders in 'Tuning Controls' window | keys: [s]=save  [r]=reset  [q]=quit")

    # FPS smoothing
    t_prev = time.time()
    fps_ema = 0.0
    ema_alpha = 0.1

    while True:
        # 1) Acquire
        img_l_rgb = get_camera_image(client, CAM_LEFT)
        img_r_rgb = get_camera_image(client, CAM_RIGHT)

        # 2) Params from UI
        tp = read_trackbar_params()

        # 3) Process
        out_l = apply_tuning(img_l_rgb, tp, gamma_cache) if img_l_rgb is not None else None
        out_r = apply_tuning(img_r_rgb, tp, gamma_cache) if img_r_rgb is not None else None

        # 4) FPS calc
        t_now = time.time()
        dt = max(1e-6, t_now - t_prev)
        fps_inst = 1.0 / dt
        fps_ema = fps_ema * (1.0 - ema_alpha) + fps_inst * ema_alpha if fps_ema > 0 else fps_inst
        t_prev = t_now

        # 5) OSD + Show
        if out_l is not None:
            put_osd(out_l, tp, fps_ema)
            cv2.imshow(WINDOW_LEFT, out_l)
        if out_r is not None:
            put_osd(out_r, tp, fps_ema)
            cv2.imshow(WINDOW_RIGHT, out_r)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Reset sliders to defaults
            defaults = TuningParams()
            cv2.setTrackbarPos("contrast x100 (0-300)", WINDOW_CTRL, defaults.contrast_x100)
            cv2.setTrackbarPos("brightness (-100..+100)", WINDOW_CTRL, defaults.brightness_off)
            cv2.setTrackbarPos("saturation x100 (0-300)", WINDOW_CTRL, defaults.saturation_x100)
            cv2.setTrackbarPos("gamma x100 (10-300)", WINDOW_CTRL, defaults.gamma_x100)
            gamma_cache["gamma_lut"] = None
            gamma_cache["gamma_val"] = None
        elif key == ord('s'):
            if h0 is None or w0 is None:
                # Try to infer from current frame if not probed
                if out_l is not None:
                    h0, w0 = out_l.shape[:2]
                elif out_r is not None:
                    h0, w0 = out_r.shape[:2]
                else:
                    h0, w0 = 0, 0
            path = save_params(tp, (w0, h0))
            print(f"Saved tuning to: {path}")

    cv2.destroyAllWindows()
    client.enableApiControl(False)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
