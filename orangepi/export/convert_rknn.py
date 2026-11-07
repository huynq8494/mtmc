"""Convert the ONNX models to RKNN (run on the x86 PC with rknn-toolkit2 installed).

  python orangepi/export/convert_rknn.py                       # fp16, both models
  python orangepi/export/convert_rknn.py --dtype int8          # needs calibration lists (make_calib_dataset.py)
  python orangepi/export/convert_rknn.py --dtype both          # fp16 and int8 side by side
  python orangepi/export/convert_rknn.py --only osnet --dtype fp16

Outputs: models/yolov8n_person_{fp16,int8}.rknn  models/osnet_x0_25_{fp16,int8}.rknn
INT8 can fail or lose accuracy (the YOLO raw head mixes DFL box logits and a person logit, OSNet embeddings are
sensitive), so fp16 is the default and both variants can be produced. Mean/std are baked into the model,
the board feeds raw uint8 RGB.
"""
import argparse, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# (mean, std) in 0..255 RGB units
NORM = {
    "yolo": ([[0, 0, 0]], [[255, 255, 255]]),
    "osnet": ([[123.675, 116.28, 103.53]], [[58.395, 57.12, 57.375]]),  # ImageNet
}
MODELS = {"yolo": "yolov8n_person", "osnet": "osnet_x0_25"}


def convert(kind, onnx, out, dtype, dataset, target, algo):
    from rknn.api import RKNN
    quant = dtype == "int8"
    if quant and not (dataset and Path(dataset).exists()):
        raise SystemExit(f"int8 needs a calibration list for {kind}: {dataset} (see make_calib_dataset.py)")
    mean, std = NORM[kind]
    r = RKNN(verbose=False)
    cfg = dict(mean_values=mean, std_values=std, target_platform=target)
    if quant:
        cfg["quantized_algorithm"] = algo
    r.config(**cfg)
    if r.load_onnx(model=str(onnx)) != 0:
        raise SystemExit(f"load_onnx failed: {onnx}")
    if r.build(do_quantization=quant, dataset=str(dataset) if quant else None) != 0:
        raise SystemExit(f"build failed ({kind} {dtype})")
    if r.export_rknn(str(out)) != 0:
        raise SystemExit(f"export_rknn failed: {out}")
    r.release()
    print(f"  wrote {out} ({Path(out).stat().st_size / 1e6:.1f} MB)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", choices=["yolo", "osnet"])
    p.add_argument("--dtype", choices=["fp16", "int8", "both"], default="fp16")
    p.add_argument("--target", default="rk3588", help="OrangePi 5 Pro = rk3588")
    p.add_argument("--models-dir", default=str(ROOT / "models"))
    p.add_argument("--yolo-dataset", default=str(ROOT / "models/calib/yolo_list.txt"))
    p.add_argument("--reid-dataset", default=str(ROOT / "models/calib/reid_list.txt"))
    p.add_argument("--algo", default="normal", choices=["normal", "mmse", "kl_divergence"],
                   help="INT8 quantization algorithm (try mmse if accuracy is bad)")
    a = p.parse_args()
    d = Path(a.models_dir)
    dtypes = ["fp16", "int8"] if a.dtype == "both" else [a.dtype]
    ds = {"yolo": a.yolo_dataset, "osnet": a.reid_dataset}
    for kind in ([a.only] if a.only else ["yolo", "osnet"]):
        onnx = d / f"{MODELS[kind]}.onnx"
        if not onnx.exists():
            sys.exit(f"{onnx} not found - run export_onnx.py first")
        for dt in dtypes:
            print(f"[{kind} {dt}]")
            try:
                convert(kind, onnx, d / f"{MODELS[kind]}_{dt}.rknn", dt, ds[kind], a.target, a.algo)
            except SystemExit as e:
                if a.dtype == "both" and dt == "int8":
                    print(f"  int8 skipped: {e}")  # keep the fp16 result
                else:
                    raise


if __name__ == "__main__":
    main()
