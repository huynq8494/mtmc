"""Export the MTMC models to ONNX (run on the PC, needs torch + ultralytics).

  python orangepi/export/export_onnx.py                 # both models -> models/*.onnx
  python orangepi/export/export_onnx.py --only yolo --imgsz 544 960

YOLO11n : input 1x3xHxW (RGB, 0..1), output [1,5,N] = (cx, cy, w, h, person_score)
          (the 80 COCO class scores are cut down to class 0 = person)
OSNet   : input 1x3x256x128 (RGB, ImageNet normalised), output [1,512]
Batch size is fixed to 1 because RKNN handles dynamic batch poorly.
"""
import argparse, sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


class YoloPerson(torch.nn.Module):
    """Wrap the Ultralytics detection model and keep only box + person score."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        y = self.m(x)
        if isinstance(y, (list, tuple)):
            y = y[0]
        return torch.cat([y[:, :4], y[:, 4:5]], 1)


def _onnx_export(model, dummy, out, opset, in_name, out_name):
    kw = dict(opset_version=opset, input_names=[in_name], output_names=[out_name])
    try:
        torch.onnx.export(model, dummy, str(out), dynamo=False, **kw)  # torch >= 2.5: use the legacy exporter
    except TypeError:
        torch.onnx.export(model, dummy, str(out), **kw)
    try:  # optional simplification
        import onnx
        from onnxsim import simplify
        m, ok = simplify(onnx.load(str(out)))
        if ok:
            onnx.save(m, str(out))
    except Exception as e:  # noqa
        print(f"  (onnx-simplifier skipped: {e})")
    print(f"  wrote {out}")


def export_yolo(weights, out, hw, opset):
    from ultralytics import YOLO
    net = YOLO(weights).model.float().eval()
    net.fuse(verbose=False)
    for p in net.parameters():
        p.requires_grad = False
    net.model[-1].export = True
    dummy = torch.zeros(1, 3, *hw)
    with torch.no_grad():
        print("  yolo output shape:", tuple(YoloPerson(net)(dummy).shape))
    _onnx_export(YoloPerson(net), dummy, out, opset, "images", "output0")


def export_osnet(weights, out, opset):
    from osnet import osnet_x0_25
    net = osnet_x0_25(num_classes=4101, pretrained=False)
    net.load_state_dict(torch.load(weights, map_location="cpu"))
    net.eval()
    dummy = torch.zeros(1, 3, 256, 128)
    with torch.no_grad():
        print("  osnet output shape:", tuple(net(dummy).shape))
    _onnx_export(net, dummy, out, opset, "images", "embedding")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", choices=["yolo", "osnet"], help="export just one model")
    p.add_argument("--yolo-weights", default=str(ROOT / "yolo11n.pt"))
    p.add_argument("--osnet-weights", default=str(ROOT / "models/osnet_x0_25_msmt17.pth"))
    p.add_argument("--imgsz", type=int, nargs=2, default=[544, 960], metavar=("H", "W"),
                   help="YOLO input size (multiple of 32); 544x960 = 1920x1080 letterboxed to 960")
    p.add_argument("--out-dir", default=str(ROOT / "models"))
    p.add_argument("--opset", type=int, default=12)
    a = p.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if a.only in (None, "yolo"):
        print("[yolo11n person]")
        export_yolo(a.yolo_weights, out / "yolo11n_person.onnx", tuple(a.imgsz), a.opset)
    if a.only in (None, "osnet"):
        print("[osnet_x0_25]")
        export_osnet(a.osnet_weights, out / "osnet_x0_25.onnx", a.opset)


if __name__ == "__main__":
    main()
