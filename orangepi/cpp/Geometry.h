#pragma once
// Camera poses, ground-plane projection and the tuning constants (port of geometry.py + config.py).
#include <opencv2/core.hpp>

#include <map>
#include <string>

struct Params {
    double maxRange = 80.0;   // ignore ground hits farther than this (m)
    double fuseD = 0.8;       // cross-camera fusion distance (m), grows with range
    double gate = 1.5;        // tracker association gate (m)
    double maxMissS = 3.0;    // delete a track after this many seconds without detections
    int people = 10;          // max number of tracks
    int confirm = 3;          // frames before a track is shown (or seen by 2+ cameras at once)
};

struct Camera {
    int id = 0;
    cv::Vec3d pos;
    cv::Matx33d R;            // camera -> world, camera looks along local -Z, +Y up
    cv::Vec3d fwd, up;
};

class World {
public:
    // Loads configs/cameras.txt written by orangepi/export/export_cameras.py
    static World load(const std::string& path);

    std::map<int, Camera> cams;
    int W = 1920, H = 1080;
    double F = 1000, HFOV = 60;
    double xmin = 0, xmax = 0, ymin = 0, ymax = 0;
    Params P;

    void setHfov(double deg);
    // pixel -> ground (z=0) world xy; false if the ray does not hit the ground within maxRange
    bool ground(const Camera& c, double u, double v, cv::Point2d& xy, double& range) const;
    bool inside(const cv::Point2d& p, double margin = 0.0) const;
    cv::Point2d clamp(const cv::Point2d& p) const;
    void print(const std::vector<int>& ids) const;
};
