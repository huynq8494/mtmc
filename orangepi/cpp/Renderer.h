#pragma once
// 2D map drawing (port of the map code in main.py)
#include <opencv2/core.hpp>

#include <vector>

#include "Geometry.h"
#include "Tracker.h"

class Renderer {
public:
    Renderer(const World& w, const std::vector<int>& cam_ids);
    cv::Size size() const { return base_.size(); }
    cv::Mat draw(const std::vector<TrackView>& tracks, int frame, double t, int ncams) const;

private:
    cv::Point pt(const cv::Point2d& p) const;
    const World& w_;
    static constexpr int S = 15, M = 40;  // px per meter, margin px
    cv::Mat base_;
    std::vector<cv::Scalar> col_;
};
