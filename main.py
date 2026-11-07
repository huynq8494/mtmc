"""Multi-target multi-camera tracking on a 2D warehouse map.

  uv run main.py --check                         # camera pose sanity check only
  uv run main.py --cams 1,7,8,10,11 --device cpu # subset of cameras on CPU
  uv run main.py --debug --show                  # raw per-camera points + camera grid video
  uv run main.py --parse-cameras --check         # re-parse configs/camera_properties.txt first
"""
import argparse, time
import cv2, numpy as np

p = argparse.ArgumentParser()
p.add_argument("--parse-cameras", action="store_true", help="regenerate configs/cameras.py from the camera properties first")
p.add_argument("--camera-props", default="configs/camera_properties.txt", help="Isaac Sim camera export (used with --parse-cameras)")
p.add_argument("--camera-defaults", default="configs/camera_defaults.json", help="defaults for missing properties (used with --parse-cameras)")
p.add_argument("--cams", default="1-12", help="e.g. 1-12 or 1,7,8,10,11")
p.add_argument("--device", default="auto", help="auto|cpu|cuda")
p.add_argument("--step", type=int, default=5, help="process every Nth frame (video is 50 fps)")
p.add_argument("--max-frames", type=int, default=0, help="stop after N processed frames (0=all)")
p.add_argument("--imgsz", type=int, default=960)
p.add_argument("--hfov", type=float, help="override horizontal FOV (deg)")
p.add_argument("--out", default="out.mp4")
p.add_argument("--show", action="store_true", help="live window")
p.add_argument("--debug", action="store_true", help="draw raw points; save debug_cams.mp4")
p.add_argument("--check", action="store_true", help="print camera pose check and exit")
a = p.parse_args()

if a.parse_cameras:  # must run before 'import config', which loads configs/cameras.py
    from parse_camera_config import generate
    generate(a.camera_props, a.camera_defaults, "configs/cameras.py")
import config as C
from geometry import Camera, check
from tracker import Tracker, fuse

if a.hfov:
    C.HFOV = a.hfov
    C.F = C.W / 2 / np.tan(np.radians(a.hfov) / 2)
ids = [int(x) for s in a.cams.split(",") for x in (range(int(s.split("-")[0]), int(s.split("-")[1]) + 1) if "-" in s else [s])]
if a.check:
    check(ids); raise SystemExit

from detector import Detector
det, cams = Detector(a.device, a.imgsz), {i: Camera(i) for i in ids}
caps = {i: cv2.VideoCapture(f"videos/camera{i:02d}.mp4") for i in ids}
fps = caps[ids[0]].get(cv2.CAP_PROP_FPS)
tracker = Tracker()

# ---- map drawing -----------------------------------------------------------
S, M = 15, 40  # px per meter, margin px
MW, MH = int((C.XMAX - C.XMIN) * S) + 2 * M, int((C.YMAX - C.YMIN) * S) + 2 * M
pt = lambda xy: (int(M + (xy[0] - C.XMIN) * S), int(M + (C.YMAX - xy[1]) * S))  # y up
COL = [tuple(int(c) for c in cv2.cvtColor(np.uint8([[[h, 220, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]) for h in range(0, 180, 18)]


def base_map():
    m = np.full((MH, MW, 3), 30, np.uint8)
    cv2.rectangle(m, pt((C.XMIN, C.YMAX)), pt((C.XMAX, C.YMIN)), (90, 90, 90), 2)
    for x in range(int(C.XMIN), int(C.XMAX) + 1, 5):
        cv2.line(m, pt((x, C.YMIN)), pt((x, C.YMAX)), (45, 45, 45), 1)
    for y in range(int(C.YMIN), int(C.YMAX) + 1, 5):
        cv2.line(m, pt((C.XMIN, y)), pt((C.XMAX, y)), (45, 45, 45), 1)
    for c in cams.values():  # camera icon + FOV wedge
        h = np.degrees(np.arctan2(c.fwd[1], c.fwd[0]))
        cv2.ellipse(m, pt(c.pos), (S * 4, S * 4), 0, -h - C.HFOV / 2, -h + C.HFOV / 2, (70, 70, 110), 1)
        cv2.circle(m, pt(c.pos), 5, (0, 200, 255), -1)
        cv2.putText(m, str(c.id), (pt(c.pos)[0] + 6, pt(c.pos)[1] - 6), 0, 0.5, (0, 200, 255), 1)
    return m


BASE = base_map()


def draw(tracks, raw, n, t):
    m = BASE.copy()
    if a.debug:
        for cam, xy in raw:
            cv2.circle(m, pt(xy), 3, (120, 120, 120), -1)
    for tr in tracks:
        col = COL[(tr.id - 1) % len(COL)]
        pts = np.array([pt(q) for q in tr.trail[-40:]], np.int32)
        cv2.polylines(m, [pts], False, col, 2)
        cv2.circle(m, pt(tr.xy), 7, col, -1)
        cv2.putText(m, f"{tr.id}", (pt(tr.xy)[0] + 9, pt(tr.xy)[1] + 5), 0, 0.6, (255, 255, 255), 2)
    cv2.putText(m, f"t={t:5.1f}s frame={n} people={len(tracks)} cams={len(cams)}", (10, 20), 0, 0.5, (255, 255, 255), 1)
    return m


# ---- main loop -------------------------------------------------------------
w_map = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), fps / a.step, (MW, MH))
w_dbg, n, seen, rejected, oob, t0 = None, 0, set(), 0, 0, time.time()
stat = {i: [0, 0] for i in ids}  # per camera [detections, rejected]
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
    dets, raw = [], []
    for (i, f), (b, e) in zip(frames.items(), res):
        foot = np.c_[(b[:, 0] + b[:, 2]) / 2, b[:, 3]]
        xy, ok, r = cams[i].ground(foot)
        ok &= b[:, 3] < C.H - 4                          # drop boxes cut by the bottom edge
        good = ok & C.inside(xy, margin=1.0)             # ground limit = bounding box of all cameras
        rejected += int((~good).sum()); stat[i][0] += len(b); stat[i][1] += int((~good).sum())
        for k in np.where(good)[0]:
            dets.append((xy[k], e[k], i, r[k])); raw.append((i, xy[k]))
        if a.debug:
            for k, bb in enumerate(b):
                cv2.rectangle(f, tuple(bb[:2].astype(int)), tuple(bb[2:4].astype(int)), (0, 255, 0) if good[k] else (0, 0, 255), 4)
            cv2.putText(f, f"cam{i}", (20, 80), 0, 3, (0, 255, 255), 6)
    tracks = tracker.step(fuse(dets), a.step / fps)
    oob += int((~C.inside([t.xy for t in tracks])).sum()) if tracks else 0
    seen |= {t.id for t in tracks}
    m = draw(tracks, raw, n, n * a.step / fps)
    w_map.write(m)
    if a.debug:
        th = [cv2.resize(f, (480, 270)) for f in frames.values()]
        th += [np.zeros_like(th[0])] * (-len(th) % 4)
        g = np.vstack([np.hstack(th[k:k + 4]) for k in range(0, len(th), 4)])
        if w_dbg is None:
            w_dbg = cv2.VideoWriter("debug_cams.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps / a.step, g.shape[1::-1])
        w_dbg.write(g)
    if a.show:
        cv2.imshow("mtmc", m)
        if cv2.waitKey(1) == ord("q"): break
    n += 1
    print(f"\rframe {n} dets={len(dets)} tracks={len(tracks)} ids={len(seen)} {(time.time() - t0) / n:.2f}s/frame", end="", flush=True)

w_map.release()
print(f"\nprocessed={n} unique_ids={sorted(seen)} raw_points_rejected_outside_ground={rejected} track_positions_outside_ground={oob}")
print('per-camera [dets, rejected]:', stat)
assert oob == 0, "a track left the ground area"
