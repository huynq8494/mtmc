"""Build INT8 calibration sets from the camera videos (PC, needs ultralytics).

  python orangepi/export/make_calib_dataset.py --n 150

Writes models/calib/yolo/*.jpg (letterboxed frames, model input size), models/calib/reid/*.jpg (128x256
person crops) and the list files models/calib/yolo_list.txt / reid_list.txt used by convert_rknn.py.
rknn-toolkit2 does NOT letterbox calibration images itself, so they are stored at the exact input size.
"""
import argparse, sys
from pathlib import Path

import cv2, numpy as np

ROOT = Path(__file__).resolve().parents[2]


def letterbox(img, hw, color=114):
    h, w = img.shape[:2]
    r = min(hw[0] / h, hw[1] / w)
    nh, nw = round(h * r), round(w * r)
    out = np.full((*hw, 3), color, np.uint8)
    top, left = (hw[0] - nh) // 2, (hw[1] - nw) // 2
    out[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--videos", default=str(ROOT / "videos"))
    p.add_argument("--out", default=str(ROOT / "models/calib"))
    p.add_argument("--n", type=int, default=150, help="number of frames (spread over all videos)")
    p.add_argument("--imgsz", type=int, nargs=2, default=[544, 960], metavar=("H", "W"))
    p.add_argument("--yolo-weights", default=str(ROOT / "yolo11n.pt"))
    p.add_argument("--max-crops", type=int, default=300)
    a = p.parse_args()

    from ultralytics import YOLO
    yolo = YOLO(a.yolo_weights)
    vids = sorted(Path(a.videos).glob("*.mp4"))
    if not vids:
        sys.exit(f"no videos in {a.videos}")
    out = Path(a.out)
    (out / "yolo").mkdir(parents=True, exist_ok=True)
    (out / "reid").mkdir(parents=True, exist_ok=True)
    per_vid = max(1, a.n // len(vids))
    yl, rl, ncrop = [], [], 0
    for v in vids:
        cap = cv2.VideoCapture(str(v))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for k, idx in enumerate(np.linspace(0, max(total - 1, 0), per_vid + 2, dtype=int)[1:-1]):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, f = cap.read()
            if not ok:
                continue
            fn = out / "yolo" / f"{v.stem}_{k:03d}.jpg"
            cv2.imwrite(str(fn), letterbox(f, a.imgsz))
            yl.append(str(fn.resolve()))
            if ncrop >= a.max_crops:
                continue
            r = yolo.predict(f, classes=[0], conf=0.35, imgsz=960, verbose=False)[0]
            for j, b in enumerate(r.boxes.xyxy.cpu().numpy()):
                crop = f[int(max(b[1], 0)):int(b[3]), int(max(b[0], 0)):int(b[2])]
                if crop.size == 0:
                    continue
                cf = out / "reid" / f"{v.stem}_{k:03d}_{j}.jpg"
                cv2.imwrite(str(cf), cv2.resize(crop, (128, 256)))  # BGR on disk, toolkit converts to RGB
                rl.append(str(cf.resolve()))
                ncrop += 1
        cap.release()
    (out / "yolo_list.txt").write_text("\n".join(yl) + "\n")
    (out / "reid_list.txt").write_text("\n".join(rl) + "\n")
    print(f"{len(yl)} yolo frames, {len(rl)} reid crops -> {out}")


if __name__ == "__main__":
    main()
