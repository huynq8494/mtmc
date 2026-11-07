"""Export the MTMC models to ONNX (run on the PC, needs torch + ultralytics).

  python orangepi/export/export_onnx.py                 # both models -> models/*.onnx
  python orangepi/export/export_onnx.py --only yolo --imgsz 544 960

YOLOv8n : input 1x3xHxW (RGB, 0..1), outputs p8,p16,p32 = [1,65,H/s,W/s] raw head (64 DFL box logits + 1 person logit);
          decode/NMS on the host (rknn-toolkit2 cannot convert the Detect decode tail)
OSNet   : input 1x3x256x128 (RGB, ImageNet normalised), output [1,512]
Batch size is fixed to 1 because RKNN handles dynamic batch poorly.
"""
import argparse, sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _onnx_export(model, dummy, out, opset, in_name, out_names):
    out_names = [out_names] if isinstance(out_names, str) else list(out_names)
    kw = dict(opset_version=opset, input_names=[in_name], output_names=out_names)
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


def _raw_person_head(head):
    """Replace Detect.forward: per scale return cat(box DFL logits (64), person logit (1)) = [1,65,H,W].
    The DFL softmax, box decode, sigmoid and NMS are left to the host (rknn-toolkit2 chokes on them).
    The class conv is pruned to class 0 (person)."""
    for seq in head.cv3:
        c = seq[-1]
        n = torch.nn.Conv2d(c.in_channels, 1, 1)
        n.weight.data, n.bias.data = c.weight.data[:1].clone(), c.bias.data[:1].clone()
        seq[-1] = n

    def forward(x):
        return tuple(torch.cat([head.cv2[i](x[i]), head.cv3[i](x[i])], 1) for i in range(head.nl))

    head.forward = forward


def export_yolo(weights, out, hw, opset):
    from ultralytics import YOLO
    net = YOLO(weights).model.float().eval()
    net.fuse(verbose=False)
    for p in net.parameters():
        p.requires_grad = False
    _raw_person_head(net.model[-1])

    class Wrap(torch.nn.Module):
        def __init__(s, m):
            super().__init__()
            s.m = m

        def forward(s, x):
            return tuple(s.m(x))

    dummy = torch.zeros(1, 3, *hw)
    with torch.no_grad():
        print("  yolo output shapes:", [tuple(t.shape) for t in Wrap(net)(dummy)])
    _onnx_export(Wrap(net), dummy, out, opset, "images", ["p8", "p16", "p32"])


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
    p.add_argument("--yolo-weights", default=str(ROOT / "yolov8n.pt"))
    p.add_argument("--osnet-weights", default=str(ROOT / "models/osnet_x0_25_msmt17.pth"))
    p.add_argument("--imgsz", type=int, nargs=2, default=[544, 960], metavar=("H", "W"),
                   help="YOLO input size (multiple of 32); 544x960 = 1920x1080 letterboxed to 960")
    p.add_argument("--out-dir", default=str(ROOT / "models"))
    p.add_argument("--opset", type=int, default=12)
    a = p.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if a.only in (None, "yolo"):
        print("[yolov8n person]")
        export_yolo(a.yolo_weights, out / "yolov8n_person.onnx", tuple(a.imgsz), a.opset)
    if a.only in (None, "osnet"):
        print("[osnet_x0_25]")
        export_osnet(a.osnet_weights, out / "osnet_x0_25.onnx", a.opset)


if __name__ == "__main__":
    main()
