#include "Renderer.h"

#include <opencv2/imgproc.hpp>

#include <cmath>
#include <cstdio>

cv::Point Renderer::pt(const cv::Point2d& p) const {
    return {static_cast<int>(M + (p.x - w_.xmin) * S), static_cast<int>(M + (w_.ymax - p.y) * S)};  // y up
}

Renderer::Renderer(const World& w, const std::vector<int>& ids) : w_(w) {
    for (int h = 0; h < 180; h += 18) {
        cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(h, 220, 255)), bgr;
        cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
        const cv::Vec3b c = bgr.at<cv::Vec3b>(0, 0);
        col_.emplace_back(c[0], c[1], c[2]);
    }
    const int mw = static_cast<int>((w.xmax - w.xmin) * S) + 2 * M;
    const int mh = static_cast<int>((w.ymax - w.ymin) * S) + 2 * M;
    base_ = cv::Mat(mh, mw, CV_8UC3, cv::Scalar(30, 30, 30));
    cv::rectangle(base_, pt({w.xmin, w.ymax}), pt({w.xmax, w.ymin}), cv::Scalar(90, 90, 90), 2);
    for (int x = static_cast<int>(w.xmin); x <= static_cast<int>(w.xmax); x += 5)
        cv::line(base_, pt({double(x), w.ymin}), pt({double(x), w.ymax}), cv::Scalar(45, 45, 45), 1);
    for (int y = static_cast<int>(w.ymin); y <= static_cast<int>(w.ymax); y += 5)
        cv::line(base_, pt({w.xmin, double(y)}), pt({w.xmax, double(y)}), cv::Scalar(45, 45, 45), 1);
    for (int id : ids) {
        const Camera& c = w.cams.at(id);
        const double h = std::atan2(c.fwd[1], c.fwd[0]) * 180.0 / CV_PI;
        const cv::Point p = pt({c.pos[0], c.pos[1]});
        cv::ellipse(base_, p, cv::Size(S * 4, S * 4), 0, -h - w.HFOV / 2, -h + w.HFOV / 2, cv::Scalar(70, 70, 110), 1);
        cv::circle(base_, p, 5, cv::Scalar(0, 200, 255), -1);
        cv::putText(base_, std::to_string(id), p + cv::Point(6, -6), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(0, 200, 255), 1);
    }
}

cv::Mat Renderer::draw(const std::vector<TrackView>& tracks, int frame, double t, int ncams) const {
    cv::Mat m = base_.clone();
    for (const auto& tr : tracks) {
        const cv::Scalar col = col_[(tr.id - 1) % col_.size()];
        std::vector<cv::Point> pts;
        for (const auto& q : tr.trail) pts.push_back(pt(q));
        if (pts.size() > 1) cv::polylines(m, pts, false, col, 2);
        const cv::Point p = pt(tr.xy);
        cv::circle(m, p, 7, col, -1);
        cv::putText(m, std::to_string(tr.id), p + cv::Point(9, 5), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    cv::Scalar(255, 255, 255), 2);
    }
    char buf[128];
    std::snprintf(buf, sizeof(buf), "t=%5.1fs frame=%d people=%zu cams=%d", t, frame, tracks.size(), ncams);
    cv::putText(m, buf, cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    return m;
}
