#pragma once
// Cross-camera fusion + Kalman / Hungarian tracker with ReID cost (port of tracker.py).
#include <opencv2/core.hpp>

#include <array>
#include <vector>

#include "Geometry.h"

using Emb = std::array<float, 512>;

struct RawDet {          // one detection of one camera, already on the ground plane
    cv::Point2d xy;
    Emb emb;
    int cam;
    double range;
};

struct Fused {           // detections of several cameras merged into one person
    cv::Point2d xy;
    Emb emb;
    std::vector<int> cams;
};

struct TrackView {       // what the caller needs to draw
    int id;
    cv::Point2d xy;
    std::vector<cv::Point2d> trail;
};

std::vector<Fused> fuse(const std::vector<RawDet>& dets, const Params& p);

// Rectangular assignment (minimum cost). Returns, for every row, the assigned column or -1.
std::vector<int> hungarian(const std::vector<std::vector<double>>& cost);

class Tracker {
public:
    explicit Tracker(const World& w) : w_(w) {}
    std::vector<TrackView> step(const std::vector<Fused>& fused, double dt);

private:
    struct Track {
        int id = 0;
        cv::Matx41d x;
        cv::Matx44d P;
        Emb emb;
        int miss = 0, hits = 1;
        std::vector<int> cams;
        bool confirmed = false;
        std::vector<cv::Point2d> trail;
        cv::Point2d xy() const { return {x(0), x(1)}; }
    };
    void predict(Track& t, double dt) const;
    void update(Track& t, const Fused& d) const;

    const World& w_;
    std::vector<Track> tracks_;
    int nextId_ = 1;
};
