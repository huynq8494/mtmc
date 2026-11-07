// MTMC (multi-target multi-camera tracking on a 2D map) on the OrangePi 5 Pro: RKNN YOLO11n + OSNet.
//
//   ./mtmc_rknn --check
//   ./mtmc_rknn --cams 1,7,8 --max-frames 50
//   ./mtmc_rknn --gst            # hardware H.264 decode (mppvideodec) via GStreamer, OpenCV must have GStreamer
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "Detector.h"
#include "Geometry.h"
#include "Renderer.h"
#include "Tracker.h"

namespace {

struct Args {
    std::string cams = "1-12";
    std::string cameraFile = "configs/cameras.txt";
    std::string yolo = "models/yolo11n_person_fp16.rknn";
    std::string reid = "models/osnet_x0_25_fp16.rknn";
    std::string videos = "videos";
    std::string out = "out_rknn_cpp.mp4";
    int step = 5;
    int maxFrames = 0;
    int npuCores = 3;
    float conf = 0.35f;
    double hfov = 0;
    bool show = false, check = false, gst = false;
};

void usage() {
    std::printf(
        "usage: mtmc_rknn [options]   (run from the mtmc/ project root, or give absolute paths)\n"
        "  --cams 1-12|1,7,8     cameras to use (default 1-12)\n"
        "  --camera-file F       default configs/cameras.txt (python orangepi/export/export_cameras.py)\n"
        "  --yolo F --reid F     .rknn models (default models/*_fp16.rknn)\n"
        "  --videos DIR          directory with cameraNN.mp4\n"
        "  --step N              process every Nth frame (default 5)\n"
        "  --max-frames N        stop after N processed frames\n"
        "  --npu-cores N         NPU cores to use, 1-3 (default 3)\n"
        "  --conf X              detection threshold (default 0.35)\n"
        "  --hfov DEG            override horizontal FOV\n"
        "  --out F               output mp4 of the map (default out_rknn_cpp.mp4)\n"
        "  --gst                 decode with GStreamer + mppvideodec (hardware)\n"
        "  --show                live window\n"
        "  --check               print camera pose check and exit\n");
}

std::vector<int> parseCams(const std::string& s) {
    std::vector<int> ids;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        const size_t d = tok.find('-');
        if (d == std::string::npos) {
            ids.push_back(std::stoi(tok));
        } else {
            for (int i = std::stoi(tok.substr(0, d)); i <= std::stoi(tok.substr(d + 1)); ++i) ids.push_back(i);
        }
    }
    return ids;
}

bool parse(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto val = [&]() -> const char* {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", k.c_str()); std::exit(2); }
            return argv[++i];
        };
        if (k == "--cams") a.cams = val();
        else if (k == "--camera-file") a.cameraFile = val();
        else if (k == "--yolo") a.yolo = val();
        else if (k == "--reid") a.reid = val();
        else if (k == "--videos") a.videos = val();
        else if (k == "--out") a.out = val();
        else if (k == "--step") a.step = std::atoi(val());
        else if (k == "--max-frames") a.maxFrames = std::atoi(val());
        else if (k == "--npu-cores") a.npuCores = std::atoi(val());
        else if (k == "--conf") a.conf = static_cast<float>(std::atof(val()));
        else if (k == "--hfov") a.hfov = std::atof(val());
        else if (k == "--gst") a.gst = true;
        else if (k == "--show") a.show = true;
        else if (k == "--check") a.check = true;
        else if (k == "-h" || k == "--help") return false;
        else { std::fprintf(stderr, "unknown option %s\n", k.c_str()); return false; }
    }
    return a.step >= 1;
}

bool openVideo(cv::VideoCapture& cap, const std::string& path, bool gst) {
    if (!gst) return cap.open(path, cv::CAP_FFMPEG);
    const std::string p = "filesrc location=" + path +
                          " ! qtdemux ! h264parse ! mppvideodec ! videoconvert ! video/x-raw,format=BGR"
                          " ! appsink drop=false sync=false max-buffers=2";
    return cap.open(p, cv::CAP_GSTREAMER);
}

}  // namespace

int main(int argc, char** argv) {
    Args a;
    if (!parse(argc, argv, a)) { usage(); return 2; }

    try {
        World world = World::load(a.cameraFile);
        if (a.hfov > 0) world.setHfov(a.hfov);
        const std::vector<int> ids = parseCams(a.cams);
        for (int id : ids)
            if (!world.cams.count(id)) throw std::runtime_error("camera " + std::to_string(id) + " not in " + a.cameraFile);
        if (a.check) { world.print(ids); return 0; }

        Detector det(a.yolo, a.reid, a.conf, 0.45f, a.npuCores);
        std::vector<cv::VideoCapture> caps(ids.size());
        for (size_t k = 0; k < ids.size(); ++k) {
            char name[64];
            std::snprintf(name, sizeof(name), "/camera%02d.mp4", ids[k]);
            const std::string path = a.videos + name;
            if (!openVideo(caps[k], path, a.gst)) throw std::runtime_error("cannot open " + path);
        }
        double fps = caps[0].get(cv::CAP_PROP_FPS);
        if (fps <= 0 || fps > 240) fps = 50;

        Renderer renderer(world, ids);
        Tracker tracker(world);
        cv::VideoWriter writer(a.out, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps / a.step, renderer.size());
        if (!writer.isOpened()) throw std::runtime_error("cannot open output " + a.out);

        int n = 0, rejected = 0, oob = 0;
        std::set<int> seen;
        const auto t0 = std::chrono::steady_clock::now();
        std::vector<cv::Mat> frames(ids.size());
        while (true) {
            bool ok = true;
            for (size_t k = 0; k < ids.size() && ok; ++k) {
                for (int s = 0; s < a.step - 1; ++s) caps[k].grab();
                ok = caps[k].read(frames[k]) && !frames[k].empty();
            }
            if (!ok || (a.maxFrames && n >= a.maxFrames)) break;

            const auto res = det(frames);

            std::vector<RawDet> raws;
            for (size_t k = 0; k < ids.size(); ++k) {
                const Camera& cam = world.cams.at(ids[k]);
                for (const Det& d : res[k]) {
                    cv::Point2d xy;
                    double r;
                    const bool hit = world.ground(cam, (d.x1 + d.x2) / 2.0, d.y2, xy, r);
                    const bool good = hit && d.y2 < world.H - 4 && world.inside(xy, 1.0);
                    if (!good) { ++rejected; continue; }
                    raws.push_back({xy, d.emb, ids[k], r});
                }
            }
            const auto tracks = tracker.step(fuse(raws, world.P), static_cast<double>(a.step) / fps);
            for (const auto& t : tracks) {
                if (!world.inside(t.xy)) ++oob;
                seen.insert(t.id);
            }
            const cv::Mat m = renderer.draw(tracks, n, n * a.step / fps, static_cast<int>(ids.size()));
            writer.write(m);
            if (a.show) {
                cv::imshow("mtmc", m);
                if (cv::waitKey(1) == 'q') break;
            }
            ++n;
            const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::printf("\rframe %d dets=%zu tracks=%zu ids=%zu %.2fs/frame (yolo %.0fms reid %.0fms, summed over cores)",
                        n, raws.size(), tracks.size(), seen.size(), el / n, det.msYolo / n, det.msReid / n);
            std::fflush(stdout);
        }
        std::printf("\nprocessed=%d unique_ids=%zu raw_points_rejected=%d track_positions_outside_ground=%d\n", n,
                    seen.size(), rejected, oob);
        if (oob) { std::fprintf(stderr, "a track left the ground area\n"); return 1; }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "\nerror: %s\n", e.what());
        return 1;
    }
    return 0;
}
