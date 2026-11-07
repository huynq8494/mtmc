"""Write the camera poses/intrinsics (configs/cameras.py) as a flat text file for the C++ app.

  python orangepi/export/export_cameras.py            # -> configs/cameras.txt

Line format (one camera per line, '#' = comment):
  id  px py pz  r00 r01 r02 r10 r11 r12 r20 r21 r22  width height focal_px  hfov_deg
R is the camera->world rotation (camera looks along local -Z, +Y up).
Regenerate cameras.py first with parse_camera_config.py if camera_properties.txt changed.
"""
import argparse, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-o", "--output", default=str(ROOT / "configs/cameras.txt"))
    a = p.parse_args()
    from configs.cameras import CAMERAS
    lines = ["# id px py pz r00 r01 r02 r10 r11 r12 r20 r21 r22 width height focal_px hfov_deg"]
    for i in sorted(CAMERAS):
        c = CAMERAS[i]
        vals = [*c["position"], *(v for row in c["rotation"] for v in row)]
        w, h = c["resolution"]
        lines.append(f"{i} " + " ".join(f"{v:.9g}" for v in vals) + f" {w} {h} {c['focal_px']:.9g} {c['hfov_deg']:.9g}")
    Path(a.output).write_text("\n".join(lines) + "\n")
    print(f"{len(CAMERAS)} cameras -> {a.output}")


if __name__ == "__main__":
    main()
