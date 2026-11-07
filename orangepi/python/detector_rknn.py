"""RKNN (RK3588 NPU) replacement for detector.Detector. Needs only numpy + cv2 + rknn-toolkit-lite2.

Same call interface as the torch Detector:  det(frames) -> [(boxes_with_conf (N,5), emb (N,512)), ...]
YOLO runs once per frame on a pool of NPU cores (one context per core, one thread per context),
OSNet runs per person crop.
"""
import time
from concurrent.futures import ThreadPoolExecutor

import cv2, numpy as np
from rknnlite.api import RKNNLite

CORE_MASKS = [RKNNLite.NPU_CORE_0, RKNNLite.NPU_CORE_1, RKNNLite.NPU_CORE_2]


def _load(path, core_mask):
    r = RKNNLite()
    if r.load_rknn(path) != 0:
        raise RuntimeError(f"load_rknn failed: {path}")
    if r.init_runtime(core_mask=core_mask) != 0:
        raise RuntimeError(f"init_runtime failed: {path}")
    return r


def letterbox(img, hw, color=114):
    """Resize keeping aspect and pad to (H, W). Returns image, scale, (pad_x, pad_y)."""
    h, w = img.shape[:2]
    s = min(hw[0] / h, hw[1] / w)
    nh, nw = round(h * s), round(w * s)
    left, top = (hw[1] - nw) // 2, (hw[0] - nh) // 2
    out = np.full((*hw, 3), color, np.uint8)
    out[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return out, s, (left, top)


class RknnDetector:
    def __init__(self, yolo_path, reid_path, imgsz=(544, 960), conf=0.35, iou=0.45, cores=3):
        """imgsz = (H, W) of the exported YOLO model. cores = number of NPU cores used for YOLO (1..3)."""
        self.hw, self.conf, self.iou = tuple(imgsz), conf, iou
        cores = max(1, min(cores, 3))
        masks = CORE_MASKS[:cores]
        self.yolo = [_load(yolo_path, m) for m in masks]
        self.reid = _load(reid_path, RKNNLite.NPU_CORE_AUTO)
        self.pool = ThreadPoolExecutor(max_workers=cores)
        self.t_yolo = self.t_reid = 0.0
        print(f"[rknn-detector] yolo={yolo_path} reid={reid_path} imgsz={self.hw} npu_cores={cores}")

    # ------------------------------------------------------------------
    def _detect(self, k, frame):
        img, s, (px, py) = letterbox(frame, self.hw)
        x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)[None]                   # 1,H,W,3 uint8 (norm is inside the model)
        out = self.yolo[k % len(self.yolo)].inference(inputs=[x], data_format=["nhwc"])[0]
        p = np.asarray(out, np.float32).reshape(5, -1)                   # cx, cy, w, h, score
        m = p[4] > self.conf
        if not m.any():
            return np.zeros((0, 5), np.float32)
        p = p[:, m]
        cx, cy, w, h, sc = p
        xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
        keep = cv2.dnn.NMSBoxes([[a, b, c - a, d - b] for a, b, c, d in xyxy.tolist()], sc.tolist(), self.conf, self.iou)
        keep = np.asarray(keep, int).reshape(-1)
        b = xyxy[keep]
        b[:, [0, 2]] = ((b[:, [0, 2]] - px) / s).clip(0, frame.shape[1] - 1)
        b[:, [1, 3]] = ((b[:, [1, 3]] - py) / s).clip(0, frame.shape[0] - 1)
        return np.c_[b, sc[keep]].astype(np.float32)

    def _embed(self, crop):
        x = cv2.cvtColor(cv2.resize(crop, (128, 256)), cv2.COLOR_BGR2RGB)[None]
        e = np.asarray(self.reid.inference(inputs=[x], data_format=["nhwc"])[0], np.float32).reshape(-1)
        return e / (np.linalg.norm(e) + 1e-12)

    def __call__(self, frames):
        t0 = time.time()
        boxes = list(self.pool.map(lambda a: self._detect(*a), enumerate(frames)))
        t1 = time.time()
        res = []
        for f, b in zip(frames, boxes):
            emb = np.zeros((len(b), 512), np.float32)
            for i, bb in enumerate(b):
                crop = f[int(max(bb[1], 0)):int(bb[3]), int(max(bb[0], 0)):int(bb[2])]
                if crop.size:
                    emb[i] = self._embed(crop)
            res.append((b, emb))
        self.t_yolo += t1 - t0
        self.t_reid += time.time() - t1
        return res

    def release(self):
        for r in self.yolo + [self.reid]:
            r.release()
        self.pool.shutdown()
