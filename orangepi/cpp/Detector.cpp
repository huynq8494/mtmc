#include "Detector.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <thread>

namespace {

double nowMs() { return cv::getTickCount() * 1000.0 / cv::getTickFrequency(); }

// Letterbox BGR -> RGB canvas (W x H), grey 114 padding. Same maths as the python version.
void letterbox(const cv::Mat& src, cv::Mat& dst_rgb, int W, int H, float& scale, int& pad_x, int& pad_y) {
    scale = std::min(static_cast<float>(H) / src.rows, static_cast<float>(W) / src.cols);
    const int nw = cvRound(src.cols * scale), nh = cvRound(src.rows * scale);
    pad_x = (W - nw) / 2;
    pad_y = (H - nh) / 2;
    dst_rgb.create(H, W, CV_8UC3);
    dst_rgb.setTo(cv::Scalar(114, 114, 114));
    cv::resize(src, dst_rgb(cv::Rect(pad_x, pad_y, nw, nh)), cv::Size(nw, nh), 0, 0, cv::INTER_LINEAR);
    cv::cvtColor(dst_rgb, dst_rgb, cv::COLOR_BGR2RGB);
}

struct Cand {
    float x1, y1, x2, y2, s;
};

std::vector<int> nms(const std::vector<Cand>& c, float iou_thr) {
    std::vector<int> idx(c.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int a, int b) { return c[a].s > c[b].s; });
    std::vector<int> keep;
    std::vector<char> dead(c.size(), 0);
    for (size_t ii = 0; ii < idx.size(); ++ii) {
        const int i = idx[ii];
        if (dead[i]) continue;
        keep.push_back(i);
        const float ai = (c[i].x2 - c[i].x1) * (c[i].y2 - c[i].y1);
        for (size_t jj = ii + 1; jj < idx.size(); ++jj) {
            const int j = idx[jj];
            if (dead[j]) continue;
            const float iw = std::min(c[i].x2, c[j].x2) - std::max(c[i].x1, c[j].x1);
            const float ih = std::min(c[i].y2, c[j].y2) - std::max(c[i].y1, c[j].y1);
            if (iw <= 0 || ih <= 0) continue;
            const float inter = iw * ih;
            const float aj = (c[j].x2 - c[j].x1) * (c[j].y2 - c[j].y1);
            if (inter / (ai + aj - inter) > iou_thr) dead[j] = 1;
        }
    }
    return keep;
}

}  // namespace

Detector::Detector(const std::string& yolo_path, const std::string& reid_path, float conf, float iou, int cores)
    : conf_(conf), iou_(iou) {
    static const int kMasks[3] = {1, 2, 4};  // RKNN_NPU_CORE_0 / 1 / 2
    cores = std::max(1, std::min(cores, 3));
    for (int i = 0; i < cores; ++i) {
        auto w = std::make_unique<Worker>();
        w->yolo.load(yolo_path, kMasks[i]);
        w->reid.load(reid_path, kMasks[i]);
        workers_.push_back(std::move(w));
    }
    std::printf("[detector] yolo=%s (%dx%d) reid=%s (%dx%d) npu_cores=%d\n", yolo_path.c_str(),
                workers_[0]->yolo.inputW(), workers_[0]->yolo.inputH(), reid_path.c_str(),
                workers_[0]->reid.inputW(), workers_[0]->reid.inputH(), cores);
}

void Detector::process(Worker& w, const cv::Mat& frame, std::vector<Det>& out) {
    // ---- YOLO ----
    double t0 = nowMs();
    const int W = w.yolo.inputW(), H = w.yolo.inputH();
    cv::Mat in;
    float s;
    int px, py;
    letterbox(frame, in, W, H, s, px, py);
    std::vector<RknnOutput> outs;
    w.yolo.run(in.data, outs);

    // YOLOv8 raw head: 3 outputs [1,65,h,w] = 64 DFL box logits (4 sides x 16 bins) + 1 person logit.
    std::vector<Cand> cand;
    for (const RknnOutput& o : outs) {
        if (o.dims.size() != 4 || o.dims[1] != 65) continue;
        const int gh = o.dims[2], gw = o.dims[3], n = gh * gw;
        const float stride = static_cast<float>(H) / gh;
        const float* p = o.data.data();
        for (int i = 0; i < n; ++i) {
            const float sc = 1.f / (1.f + std::exp(-p[64 * n + i]));
            if (sc <= conf_) continue;
            float d[4];
            for (int k = 0; k < 4; ++k) {  // softmax over 16 bins -> expected distance in cells
                float mx = -1e30f;
                for (int j = 0; j < 16; ++j) mx = std::max(mx, p[(k * 16 + j) * n + i]);
                float sum = 0.f, acc = 0.f;
                for (int j = 0; j < 16; ++j) {
                    const float e = std::exp(p[(k * 16 + j) * n + i] - mx);
                    sum += e;
                    acc += e * j;
                }
                d[k] = acc / sum;
            }
            const float gx = (i % gw) + 0.5f, gy = (i / gw) + 0.5f;
            cand.push_back({(gx - d[0]) * stride, (gy - d[1]) * stride, (gx + d[2]) * stride, (gy + d[3]) * stride, sc});
        }
    }
    const std::vector<int> keep = nms(cand, iou_);
    w.msYolo += nowMs() - t0;

    // ---- ReID on every kept box ----
    t0 = nowMs();
    cv::Mat crop_rgb;
    std::vector<RknnOutput> eo;
    const int RW = w.reid.inputW(), RH = w.reid.inputH();
    for (int k : keep) {
        const Cand& c = cand[k];
        Det d;
        d.x1 = std::min(std::max((c.x1 - px) / s, 0.f), static_cast<float>(frame.cols - 1));
        d.x2 = std::min(std::max((c.x2 - px) / s, 0.f), static_cast<float>(frame.cols - 1));
        d.y1 = std::min(std::max((c.y1 - py) / s, 0.f), static_cast<float>(frame.rows - 1));
        d.y2 = std::min(std::max((c.y2 - py) / s, 0.f), static_cast<float>(frame.rows - 1));
        d.conf = c.s;
        d.emb.fill(0.f);
        const cv::Rect r(cv::Point(static_cast<int>(d.x1), static_cast<int>(d.y1)),
                         cv::Point(static_cast<int>(d.x2), static_cast<int>(d.y2)));
        if (r.width > 1 && r.height > 1) {
            cv::resize(frame(r), crop_rgb, cv::Size(RW, RH), 0, 0, cv::INTER_LINEAR);
            cv::cvtColor(crop_rgb, crop_rgb, cv::COLOR_BGR2RGB);
            w.reid.run(crop_rgb.data, eo);
            const auto& e = eo[0].data;
            double n = 0;
            for (size_t i = 0; i < 512 && i < e.size(); ++i) n += double(e[i]) * e[i];
            n = std::sqrt(n) + 1e-12;
            for (size_t i = 0; i < 512 && i < e.size(); ++i) d.emb[i] = static_cast<float>(e[i] / n);
        }
        out.push_back(d);
    }
    w.msReid += nowMs() - t0;
}

std::vector<std::vector<Det>> Detector::operator()(const std::vector<cv::Mat>& frames) {
    std::vector<std::vector<Det>> res(frames.size());
    const size_t nw = workers_.size();
    std::vector<std::thread> th;
    for (size_t j = 0; j < nw; ++j) {
        th.emplace_back([&, j]() {
            for (size_t k = j; k < frames.size(); k += nw) process(*workers_[j], frames[k], res[k]);
        });
    }
    for (auto& t : th) t.join();
    msYolo = msReid = 0;
    for (auto& w : workers_) { msYolo += w->msYolo; msReid += w->msReid; }
    return res;
}
