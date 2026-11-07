#include "Geometry.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>

World World::load(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) throw std::runtime_error("cannot open camera file: " + path + " (run export_cameras.py)");
    World w;
    std::string line;
    bool first = true;
    double focal0 = 0;
    int w0 = 0, h0 = 0;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        Camera c;
        double r[9], px, py, pz, focal, hfov;
        int width, height;
        ss >> c.id >> px >> py >> pz;
        for (double& v : r) ss >> v;
        ss >> width >> height >> focal >> hfov;
        if (ss.fail()) throw std::runtime_error("bad camera line: " + line);
        c.pos = cv::Vec3d(px, py, pz);
        c.R = cv::Matx33d(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8]);
        c.fwd = c.R * cv::Vec3d(0, 0, -1);
        c.up = c.R * cv::Vec3d(0, 1, 0);
        if (first) {
            w0 = width; h0 = height; focal0 = focal; w.HFOV = hfov;
        } else if (width != w0 || height != h0 || std::fabs(focal - focal0) > 1e-6) {
            throw std::runtime_error("per-camera intrinsics are not supported (camera " + std::to_string(c.id) + ")");
        }
        w.cams[c.id] = c;
        if (first) {
            w.xmin = w.xmax = px; w.ymin = w.ymax = py;
            first = false;
        } else {
            w.xmin = std::min(w.xmin, px); w.xmax = std::max(w.xmax, px);
            w.ymin = std::min(w.ymin, py); w.ymax = std::max(w.ymax, py);
        }
    }
    if (w.cams.empty()) throw std::runtime_error("no cameras in " + path);
    w.W = w0; w.H = h0; w.F = focal0;
    return w;
}

void World::setHfov(double deg) {
    HFOV = deg;
    F = W / 2.0 / std::tan(deg * CV_PI / 180.0 / 2.0);
}

bool World::ground(const Camera& c, double u, double v, cv::Point2d& xy, double& range) const {
    cv::Vec3d d((u - W / 2.0) / F, -(v - H / 2.0) / F, -1.0);
    d = c.R * d;
    d *= 1.0 / cv::norm(d);
    if (!(d[2] < -1e-3)) return false;
    const double t = -c.pos[2] / d[2];
    if (!(t < P.maxRange)) return false;
    xy = cv::Point2d(c.pos[0] + t * d[0], c.pos[1] + t * d[1]);
    range = t;
    return true;
}

bool World::inside(const cv::Point2d& p, double m) const {
    return p.x >= xmin - m && p.x <= xmax + m && p.y >= ymin - m && p.y <= ymax + m;
}

cv::Point2d World::clamp(const cv::Point2d& p) const {
    return {std::min(std::max(p.x, xmin), xmax), std::min(std::max(p.y, ymin), ymax)};
}

void World::print(const std::vector<int>& ids) const {
    std::printf("bounds x[%.1f,%.1f] y[%.1f,%.1f]  HFOV=%.1fdeg F=%.0fpx  %dx%d\n", xmin, xmax, ymin, ymax, HFOV, F, W, H);
    for (int i : ids) {
        auto it = cams.find(i);
        if (it == cams.end()) { std::printf("cam%02d: not in camera file\n", i); continue; }
        const Camera& c = it->second;
        cv::Point2d xy;
        double r;
        const bool ok = ground(c, W / 2.0, H - 1.0, xy, r);
        std::printf("cam%02d pos=(%.1f,%.1f,%.1f) fwd=(%.2f,%.2f,%.2f) up_z=%+.2f bottom_hit=%s%s\n", i, c.pos[0], c.pos[1],
                    c.pos[2], c.fwd[0], c.fwd[1], c.fwd[2], c.up[2],
                    ok ? (std::to_string(xy.x).substr(0, 5) + "," + std::to_string(xy.y).substr(0, 5)).c_str() : "none",
                    c.up[2] < 0 ? " UPSIDE-DOWN" : "");
    }
}
