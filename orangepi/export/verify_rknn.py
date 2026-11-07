"""Check an .rknn model against its ONNX original on the PC (RKNN simulator, no board needed).

  python orangepi/export/verify_rknn.py --kind yolo  --rknn models/yolo11n_person_fp16.rknn --image frame.jpg
  python orangepi/export/verify_rknn.py --kind osnet --rknn models/osnet_x0_25_int8.rknn   --image crop.jpg

yolo : compares score maps (max abs diff) and the number / position of detections above 0.35
osnet: prints the cosine similarity of the two embeddings (should be > 0.98 for fp16, > 0.9 for int8)
"""
import argparse
from pathlib import Path

import cv2, numpy as np

ROOT = Path(__file__).resolve().parents[2]
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)


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
    p.add_argument("--kind", choices=["yolo", "osnet"], required=True)
    p.add_argument("--rknn", required=True)
    p.add_argument("--onnx")
    p.add_argument("--image", required=True)
    p.add_argument("--imgsz", type=int, nargs=2, default=[544, 960], metavar=("H", "W"))
    a = p.parse_args()

    import onnxruntime as ort
    from rknn.api import RKNN
    onnx = a.onnx or str(ROOT / "models" / ("yolo11n_person.onnx" if a.kind == "yolo" else "osnet_x0_25.onnx"))
    img = cv2.imread(a.image)
    assert img is not None, f"cannot read {a.image}"
    if a.kind == "yolo":
        x = cv2.cvtColor(letterbox(img, tuple(a.imgsz)), cv2.COLOR_BGR2RGB)
        ref_in = (x.astype(np.float32) / 255).transpose(2, 0, 1)[None]
    else:
        x = cv2.cvtColor(cv2.resize(img, (128, 256)), cv2.COLOR_BGR2RGB)
        ref_in = ((x.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1)[None]

    sess = ort.InferenceSession(onnx, providers=["CPUExecutionProvider"])
    ref = sess.run(None, {sess.get_inputs()[0].name: ref_in})[0]

    r = RKNN(verbose=False)
    assert r.load_rknn(a.rknn) == 0, "load_rknn failed"
    assert r.init_runtime() == 0, "init_runtime (simulator) failed"
    out = r.inference(inputs=[x[None]], data_format=["nhwc"])[0]
    r.release()

    print("onnx shape:", ref.shape, " rknn shape:", out.shape)
    if a.kind == "osnet":
        u, v = ref.ravel(), out.ravel()
        print(f"cosine similarity: {u @ v / np.linalg.norm(u) / np.linalg.norm(v):.4f}")
        return
    ref, out = ref.reshape(5, -1), out.reshape(5, -1)
    print(f"max |score diff|: {np.abs(ref[4] - out[4]).max():.3f}   max |box diff| (px): {np.abs(ref[:4] - out[:4]).max():.2f}")
    rk, ok_ = np.where(ref[4] > 0.35)[0], np.where(out[4] > 0.35)[0]
    print(f"candidates > 0.35: onnx={len(rk)} rknn={len(ok_)}")
    for i in ok_[np.argsort(-out[4][ok_])][:5]:
        print(f"  rknn  conf={out[4][i]:.2f} cx,cy,w,h={np.round(out[:4, i], 1)}")


if __name__ == "__main__":
    main()
