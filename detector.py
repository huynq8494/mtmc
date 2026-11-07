import cv2, numpy as np, torch
from ultralytics import YOLO
from osnet import osnet_x0_25
import config as C

MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)


class Detector:
    def __init__(self, device="auto", imgsz=960, conf=0.35):
        self.dev = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        self.imgsz, self.conf = imgsz, conf
        self.yolo = YOLO("yolo11n.pt")
        self.reid = osnet_x0_25(num_classes=4101, pretrained=False)
        self.reid.load_state_dict(torch.load("models/osnet_x0_25_msmt17.pth", map_location="cpu"))
        self.reid.eval().to(self.dev)
        print(f"[detector] device={self.dev} imgsz={imgsz}")

    @torch.no_grad()
    def __call__(self, frames):
        """frames: list of BGR images -> per frame list of (x1,y1,x2,y2,conf) and (N,512) embeddings."""
        res = self.yolo.predict(frames, classes=[0], conf=self.conf, imgsz=self.imgsz,
                                device=self.dev, verbose=False, **({"half": True} if self.dev == "cuda" else {}))
        boxes = [r.boxes.xyxy.cpu().numpy() for r in res]
        confs = [r.boxes.conf.cpu().numpy() for r in res]
        crops = [cv2.resize(f[int(max(b[1], 0)):int(b[3]), int(max(b[0], 0)):int(b[2])][..., ::-1], (128, 256))
                 for f, bs in zip(frames, boxes) for b in bs]
        emb = np.zeros((0, 512), np.float32)
        if crops:
            x = (np.stack(crops).astype(np.float32) / 255 - MEAN) / STD
            e = self.reid(torch.from_numpy(x).permute(0, 3, 1, 2).to(self.dev)).float()
            emb = torch.nn.functional.normalize(e, dim=1).cpu().numpy()
        out, k = [], 0
        for b, c in zip(boxes, confs):
            out.append((np.c_[b, c], emb[k:k + len(b)]))
            k += len(b)
        return out
