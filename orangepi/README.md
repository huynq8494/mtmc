# MTMC on OrangePi 5 Pro (RK3588 NPU, RKNN)

Pipeline is unchanged (YOLO11n person -> foot point -> ground projection -> fusion -> Kalman/Hungarian + OSNet ReID),
only the two neural networks run on the NPU.

```
orangepi/
  export/   PC side: ONNX export, RKNN conversion, calibration data, verification, camera file
  python/   board side: Python version (rknn-toolkit-lite2), reuses ../config.py ../geometry.py ../tracker.py
  cpp/      board side: C++ app (librknnrt + OpenCV), optional GStreamer hardware decode (--gst)
```

## 1. Export models (x86 PC)

Needs the project venv (torch, ultralytics) plus `rknn-toolkit2`, `onnx`, `onnxruntime`, optional `onnxsim`.
Run from `mtmc/`.

```bash
python orangepi/export/export_onnx.py                  # models/yolov8n_person.onnx (YOLOv8n person-only raw head: 1x3x544x960 -> 3 x [1,65,h,w]; DFL decode + NMS run on the host), models/osnet_x0_25.onnx (1x3x256x128 -> [1,512])
python orangepi/export/convert_rknn.py                 # FP16 (default, safest): models/*_fp16.rknn
python orangepi/export/make_calib_dataset.py --n 150   # only for INT8: models/calib/*_list.txt
python orangepi/export/convert_rknn.py --dtype int8    # models/*_int8.rknn   (or --dtype both)
python orangepi/export/verify_rknn.py --kind osnet --rknn models/osnet_x0_25_fp16.rknn --image crop.jpg
python orangepi/export/verify_rknn.py --kind yolo  --rknn models/yolov8n_person_fp16.rknn --image frame.jpg
python orangepi/export/export_cameras.py               # configs/cameras.txt (for the C++ app)
```

Notes
- Normalisation (YOLO /255, OSNet ImageNet mean/std) is baked into the RKNN, the board feeds raw uint8 RGB.
- Batch is fixed to 1. YOLO input is 544x960 (1920x1080 letterboxed to 960). For another size use
  `--imgsz H W` in `export_onnx.py`, and the same value in `main_rknn.py --imgsz`; C++ reads it from the model.
- INT8 may fail or hurt accuracy (YOLO output mixes pixel boxes and 0..1 scores; ReID embeddings are sensitive).
  Mixed use is fine, e.g. `--yolo ..._int8.rknn --reid ..._fp16.rknn`. Try `--algo mmse` if INT8 is poor.
  `verify_rknn.py` should give cosine similarity > 0.98 (fp16) for OSNet.
- Copy to the board: `models/*.rknn`, `configs/` (cameras.py/cameras.txt), `videos/`, the `orangepi/` folder and the
  top-level `config.py geometry.py tracker.py` (Python version only).

## 2. Python on the board

```bash
pip install -r orangepi/python/requirements-rknn.txt     # + the rknn-toolkit-lite2 wheel for your python
python orangepi/python/main_rknn.py --check
python orangepi/python/main_rknn.py --cams 1,7,8 --max-frames 50
python orangepi/python/main_rknn.py --yolo models/yolov8n_person_int8.rknn --npu-cores 3
```
Output: `out_rknn.mp4`. The per-frame log shows YOLO and ReID ms. YOLO runs on up to 3 NPU cores in parallel.

## 3. C++ on the board

Dependencies: `sudo apt install build-essential cmake libopencv-dev`, and the RKNN runtime
(`librknnrt.so` + `rknn_api.h` from https://github.com/airockchip/rknn-toolkit2 `rknpu2/`, or from `rknn_model_zoo/3rdparty/rknpu2`).

```bash
cd orangepi/cpp
cmake -S . -B build -DRKNN_MODEL_ZOO=/path/to/rknn_model_zoo    # or -DRKNN_API_DIR=<dir with include/ and lib/> or nothing if installed system wide
cmake --build build -j4
cd ../..                                                          # run from mtmc/ so default relative paths work
./orangepi/cpp/build/mtmc_rknn --check
./orangepi/cpp/build/mtmc_rknn --cams 1,7,8 --max-frames 50
./orangepi/cpp/build/mtmc_rknn --gst                              # GStreamer + mppvideodec hardware decode
./orangepi/cpp/build/mtmc_rknn --help
```
`--gst` needs OpenCV built with GStreamer and the Rockchip `mpp` GStreamer plugin (`gst-inspect-1.0 mppvideodec`).
Without it, OpenCV/FFmpeg decodes on the CPU (12 x 1080p streams will be slow; raise `--step`).

## Differences to the torch version
- Detection uses a fixed 544x960 letterbox, torch used `imgsz=960` (same scale). Small numeric differences are expected.
- Tracker/fusion are identical (C++ is a direct port, with its own Hungarian solver instead of scipy).
- If `LOOKS-OUTSIDE-BOUNDS` / track count differ strongly from the torch run, first compare with FP16 models.
