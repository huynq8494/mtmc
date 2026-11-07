"""MTMC on the OrangePi 5 Pro (RK3588) with RKNN models. Python version, no torch needed.

  python orangepi/python/main_rknn.py --check
  python orangepi/python/main_rknn.py --cams 1,7,8 --max-frames 50
  python orangepi/python/main_rknn.py --yolo models/yolov8n_person_int8.rknn --reid models/osnet_x0_25_fp16.rknn

Reuses config.py / geometry.py / tracker.py from the parent folder. Run from anywhere; relative paths
(videos, models, outputs) are resolved against the mtmc/ project root.
"""
import argparse, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.chdir(ROOT)  # configs/, videos/, models/ are relative to the project root

import cv2, numpy as np

p = argparse.ArgumentParser()
p.add_argument("--cams", default="1-12", help="e.g. 1-12 or 1,7,8,10,11")
p.add_argument("--yolo", default="models/yolov8n_person_fp16.rknn")
p.add_argument("--reid", default="models/osnet_x0_25_fp16.rknn")
p.add_argument("--imgsz", type=int, nargs=2, default=[544, 960], metavar=("H", "W"), help="YOLO model input size")
p.add_argument("--conf", type=float, default=0.35)
p.add_argument("--npu-cores", type=int, default=3, help="NPU cores used for YOLO (1-3)")
p.add_argument("--videos", default="videos")
p.add_argument("--step", type=int, default=5, help="process every Nth frame (video is 50 fps)")
p.add_argument("--max-frames", type=int, default=0, help="stop after N processed frames (0=all)")
p.add_argument("--hfov", type=float, help="override horizontal FOV (deg)")
p.add_argument("--out", default="out_rknn.mp4")
p.add_argument("--show", action="store_true", help="live window (needs a display)")
p.add_argument("--check", action="store_true", help="print camera pose check and exit")
a = p.parse_args()

import config as C
from geometry import Camera, check
from tracker import Tracker, fuse

if a.hfov:
    C.HFOV = a.hfov
    C.F = C.W / 2 / np.tan(np.radians(a.hfov) / 2)
ids = [int(x) for s in a.cams.split(",") for x in (range(int(s.split("-")[0]), int(s.split("-")[1]) + 1) if "-" in s else [s])]
if a.check:
    check(ids); raise SystemExit

from detector_rknn import RknnDetector
det, cams = RknnDetector(a.yolo, a.reid, a.imgsz, a.conf, cores=a.npu_cores), {i: Camera(i) for i in ids}
caps = {i: cv2.VideoCapture(f"{a.videos}/camera{i:02d}.mp4") for i in ids}
fps = caps[ids[0]].get(cv2.CAP_PROP_FPS)
tracker = Tracker()

# ---- map drawing (same as main.py) ------------------------------------------
S, M = 15, 40
MW, MH = int((C.XMAX - C.XMIN) * S) + 2 * M, int((C.YMAX - C.YMIN) * S) + 2 * M
pt = lambda xy: (int(M + (xy[0] - C.XMIN) * S), int(M + (C.YMAX - xy[1]) * S))
COL = [tuple(int(c) for c in cv2.cvtColor(np.uint8([[[h, 220, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]) for h in range(0, 180, 18)]


def base_map():
    m = np.full((MH, MW, 3), 30, np.uint8)
    cv2.rectangle(m, pt((C.XMIN, C.YMAX)), pt((C.XMAX, C.YMIN)), (90, 90, 90), 2)
    for x in range(int(C.XMIN), int(C.XMAX) + 1, 5):
        cv2.line(m, pt((x, C.YMIN)), pt((x, C.YMAX)), (45, 45, 45), 1)
    for y in range(int(C.YMIN), int(C.YMAX) + 1, 5):
        cv2.line(m, pt((C.XMIN, y)), pt((C.XMAX, y)), (45, 45, 45), 1)
    for c in cams.values():
        h = np.degrees(np.arctan2(c.fwd[1], c.fwd[0]))
        cv2.ellipse(m, pt(c.pos), (S * 4, S * 4), 0, -h - C.HFOV / 2, -h + C.HFOV / 2, (70, 70, 110), 1)
        cv2.circle(m, pt(c.pos), 5, (0, 200, 255), -1)
        cv2.putText(m, str(c.id), (pt(c.pos)[0] + 6, pt(c.pos)[1] - 6), 0, 0.5, (0, 200, 255), 1)
    return m


BASE = base_map()


def draw(tracks, n, t):
    m = BASE.copy()
    for tr in tracks:
        col = COL[(tr.id - 1) % len(COL)]
        pts = np.array([pt(q) for q in tr.trail[-40:]], np.int32)
        cv2.polylines(m, [pts], False, col, 2)
        cv2.circle(m, pt(tr.xy), 7, col, -1)
        cv2.putText(m, f"{tr.id}", (pt(tr.xy)[0] + 9, pt(tr.xy)[1] + 5), 0, 0.6, (255, 255, 255), 2)
    cv2.putText(m, f"t={t:5.1f}s frame={n} people={len(tracks)} cams={len(cams)}", (10, 20), 0, 0.5, (255, 255, 255), 1)
    return m


# ---- main loop ----------------------------------------------------------------
writer = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), fps / a.step, (MW, MH))
n, seen, rejected, oob, t0 = 0, set(), 0, 0, time.time()
try:
    while True:
        frames = {}
        for i, cap in caps.items():
            for _ in range(a.step - 1):
                cap.grab()
            ok, f = cap.read()
            if ok: frames[i] = f
        if len(frames) < len(caps) or (a.max_frames and n >= a.max_frames):
            break
        res = det(list(frames.values()))
        dets = []
        for i, (b, e) in zip(frames, res):
            foot = np.c_[(b[:, 0] + b[:, 2]) / 2, b[:, 3]]
            xy, ok, r = cams[i].ground(foot)
            ok &= b[:, 3] < C.H - 4
            good = ok & C.inside(xy, margin=1.0)
            rejected += int((~good).sum())
            for k in np.where(good)[0]:
                dets.append((xy[k], e[k], i, r[k]))
        tracks = tracker.step(fuse(dets), a.step / fps)
        oob += int((~C.inside([t.xy for t in tracks])).sum()) if tracks else 0
        seen |= {t.id for t in tracks}
        m = draw(tracks, n, n * a.step / fps)
        writer.write(m)
        if a.show:
            cv2.imshow("mtmc", m)
            if cv2.waitKey(1) == ord("q"): break
        n += 1
        print(f"\rframe {n} dets={len(dets)} tracks={len(tracks)} ids={len(seen)} {(time.time() - t0) / n:.2f}s/frame "
              f"(yolo {det.t_yolo / n * 1000:.0f}ms reid {det.t_reid / n * 1000:.0f}ms)", end="", flush=True)
finally:
    writer.release()
    det.release()
print(f"\nprocessed={n} unique_ids={sorted(seen)} raw_points_rejected={rejected} track_positions_outside_ground={oob}")
assert oob == 0, "a track left the ground area"
