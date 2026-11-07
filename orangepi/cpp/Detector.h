#pragma once
// YOLO11n person detector + OSNet ReID on the RK3588 NPU.
// One YOLO and one ReID context per NPU core; frames of a batch are spread over the cores (one thread each).
#include <opencv2/core.hpp>

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "RknnModel.h"

struct Det {
    float x1, y1, x2, y2, conf;
    std::array<float, 512> emb;
};

class Detector {
public:
    Detector(const std::string& yolo_path, const std::string& reid_path, float conf = 0.35f, float iou = 0.45f,
             int cores = 3);
    // frames: BGR images. Returns detections (original pixel coordinates) with L2-normalised embeddings.
    std::vector<std::vector<Det>> operator()(const std::vector<cv::Mat>& frames);

    double msYolo = 0, msReid = 0;  // accumulated thread time

private:
    struct Worker {
        RknnModel yolo, reid;
        double msYolo = 0, msReid = 0;
    };
    void process(Worker& w, const cv::Mat& frame, std::vector<Det>& out);
    std::vector<std::unique_ptr<Worker>> workers_;
    float conf_, iou_;
};
