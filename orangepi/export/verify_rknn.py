"""Check an .rknn model against its ONNX original on the PC (RKNN simulator, no board needed).

  python orangepi/export/verify_rknn.py --kind yolo  --rknn models/yolov8n_person_fp16.rknn --image frame.jpg
  python orangepi/export/verify_rknn.py --kind osnet --rknn models/osnet_x0_25_int8.rknn   --image crop.jpg

yolo : compares the raw head outputs (max abs diff) and the decoded detections above 0.35
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


def decode(outs, hw, conf=0.35):
    """Same maths as detector_rknn.decode_head (kept here so this script needs no rknnlite)."""
    B, S = [], []
    for o in outs:
        o = np.asarray(o, np.float32)
        _, c, h, w = o.shape
        st = hw[0] // h
        o = o.reshape(c, h * w)
        sc = 1 / (1 + np.exp(-o[64]))
        m = sc > conf
        if not m.any():
            continue
        d = o[:64, m].reshape(4, 16, -1)
        d = np.exp(d - d.max(1, keepdims=True))
        d = ((d / d.sum(1, keepdims=True)) * np.arange(16, dtype=np.float32)[None, :, None]).sum(1)
        idx = np.nonzero(m)[0]
        gx, gy = idx % w + 0.5, idx // w + 0.5
        B.append(np.stack([(gx - d[0]) * st, (gy - d[1]) * st, (gx + d[2]) * st, (gy + d[3]) * st], 1))
        S.append(sc[m])
    if not B:
        return np.zeros((0, 4)), np.zeros(0)
    B, S = np.concatenate(B), np.concatenate(S)
    k = cv2.dnn.NMSBoxes([[a, b, c - a, d - b] for a, b, c, d in B.tolist()], S.tolist(), conf, 0.45)
    k = np.asarray(k, int).reshape(-1)
    return B[k], S[k]


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
    onnx = a.onnx or str(ROOT / "models" / ("yolov8n_person.onnx" if a.kind == "yolo" else "osnet_x0_25.onnx"))
    img = cv2.imread(a.image)
    assert img is not None, f"cannot read {a.image}"
    if a.kind == "yolo":
        x = cv2.cvtColor(letterbox(img, tuple(a.imgsz)), cv2.COLOR_BGR2RGB)
        ref_in = (x.astype(np.float32) / 255).transpose(2, 0, 1)[None]
    else:
        x = cv2.cvtColor(cv2.resize(img, (128, 256)), cv2.COLOR_BGR2RGB)
        ref_in = ((x.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1)[None]

    sess = ort.InferenceSession(onnx, providers=["CPUExecutionProvider"])
    refs = sess.run(None, {sess.get_inputs()[0].name: ref_in})
    ref = refs[0]

    # A .rknn loaded with load_rknn cannot run in the PC simulator, so rebuild from the ONNX with the same
    # settings as convert_rknn.py (fp16 or int8 chosen from the .rknn file name) and run that.
    mean, std = ([[0, 0, 0]], [[255, 255, 255]]) if a.kind == "yolo" else ([[123.675, 116.28, 103.53]], [[58.395, 57.12, 57.375]])
    quant = "int8" in Path(a.rknn).name
    ds = ROOT / "models/calib" / ("yolo_list.txt" if a.kind == "yolo" else "reid_list.txt")
    r = RKNN(verbose=False)
    r.config(mean_values=mean, std_values=std, target_platform="rk3588")
    assert r.load_onnx(model=onnx) == 0, "load_onnx failed"
    assert r.build(do_quantization=quant, dataset=str(ds) if quant else None) == 0, "build failed"
    assert r.init_runtime() == 0, "init_runtime (simulator) failed"
    outs = r.inference(inputs=[x[None]], data_format=["nhwc"])
    out = outs[0]
    r.release()

    print("onnx shape:", ref.shape, " rknn shape:", out.shape)
    if a.kind == "osnet":
        u, v = ref.ravel(), out.ravel()
        print(f"cosine similarity: {u @ v / np.linalg.norm(u) / np.linalg.norm(v):.4f}")
        return
    print("max |raw head diff| per scale:", [f"{np.abs(u - np.asarray(v)).max():.3f}" for u, v in zip(refs, outs)])
    rb, rs = decode(refs, tuple(a.imgsz))
    ob, os_ = decode(outs, tuple(a.imgsz))
    print(f"detections > 0.35 after NMS: onnx={len(rs)} rknn={len(os_)}")
    for i in np.argsort(-os_)[:5]:
        print(f"  rknn conf={os_[i]:.2f} xyxy={np.round(ob[i], 1)}")


if __name__ == "__main__":
    main()
