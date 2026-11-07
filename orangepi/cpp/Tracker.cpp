#include "Tracker.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace {

double dot(const Emb& a, const Emb& b) {
    double s = 0;
    for (size_t i = 0; i < a.size(); ++i) s += double(a[i]) * b[i];
    return s;
}

void normalize(Emb& e) {
    const double n = std::sqrt(dot(e, e)) + 1e-12;
    for (float& v : e) v = static_cast<float>(v / n);
}

}  // namespace

// ---------------------------------------------------------------- fusion
std::vector<Fused> fuse(const std::vector<RawDet>& dets, const Params& p) {
    struct Cluster {
        cv::Point2d xy;
        std::vector<std::pair<cv::Point2d, double>> ps;
        std::vector<const Emb*> es;
        std::vector<int> cams;
    };
    std::vector<const RawDet*> order;
    for (const auto& d : dets) order.push_back(&d);
    std::stable_sort(order.begin(), order.end(), [](const RawDet* a, const RawDet* b) { return a->range < b->range; });

    std::vector<Cluster> cl;
    for (const RawDet* d : order) {
        const double thr = p.fuseD + 0.03 * d->range;
        Cluster* best = nullptr;
        double bd = thr;
        for (auto& c : cl) {
            const double dist = cv::norm(c.xy - d->xy);
            if (dist < bd && std::find(c.cams.begin(), c.cams.end(), d->cam) == c.cams.end()) {
                best = &c;
                bd = dist;
            }
        }
        const double w = 1.0 / (d->range * d->range);
        if (!best) {
            Cluster c;
            c.xy = d->xy;
            c.ps.push_back({d->xy, w});
            c.es.push_back(&d->emb);
            c.cams.push_back(d->cam);
            cl.push_back(std::move(c));
        } else {
            best->ps.push_back({d->xy, w});
            best->es.push_back(&d->emb);
            best->cams.push_back(d->cam);
            cv::Point2d s(0, 0);
            double ws = 0;
            for (const auto& q : best->ps) { s += q.first * q.second; ws += q.second; }
            best->xy = s * (1.0 / ws);
        }
    }
    std::vector<Fused> out;
    for (auto& c : cl) {
        Fused f;
        f.xy = c.xy;
        f.emb.fill(0.f);
        for (const Emb* e : c.es)
            for (size_t i = 0; i < f.emb.size(); ++i) f.emb[i] += (*e)[i] / static_cast<float>(c.es.size());
        normalize(f.emb);
        f.cams = c.cams;
        out.push_back(std::move(f));
    }
    return out;
}

// ---------------------------------------------------------------- hungarian
std::vector<int> hungarian(const std::vector<std::vector<double>>& a) {
    const int rows = static_cast<int>(a.size());
    const int cols = rows ? static_cast<int>(a[0].size()) : 0;
    std::vector<int> res(rows, -1);
    if (rows == 0 || cols == 0) return res;

    const bool tr = rows > cols;  // the algorithm needs n <= m
    const int n = tr ? cols : rows, m = tr ? rows : cols;
    auto C = [&](int i, int j) { return tr ? a[j][i] : a[i][j]; };

    const double INF = std::numeric_limits<double>::infinity();
    std::vector<double> u(n + 1, 0), v(m + 1, 0);
    std::vector<int> p(m + 1, 0), way(m + 1, 0);
    for (int i = 1; i <= n; ++i) {
        p[0] = i;
        int j0 = 0;
        std::vector<double> minv(m + 1, INF);
        std::vector<char> used(m + 1, 0);
        do {
            used[j0] = 1;
            const int i0 = p[j0];
            double delta = INF;
            int j1 = 0;
            for (int j = 1; j <= m; ++j) {
                if (used[j]) continue;
                const double cur = C(i0 - 1, j - 1) - u[i0] - v[j];
                if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
                if (minv[j] < delta) { delta = minv[j]; j1 = j; }
            }
            for (int j = 0; j <= m; ++j) {
                if (used[j]) { u[p[j]] += delta; v[j] -= delta; }
                else minv[j] -= delta;
            }
            j0 = j1;
        } while (p[j0] != 0);
        do {
            const int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0);
    }
    for (int j = 1; j <= m; ++j) {
        if (!p[j]) continue;
        if (tr) res[j - 1] = p[j] - 1;   // transposed: row index j-1 of the original, column p[j]-1
        else res[p[j] - 1] = j - 1;
    }
    return res;
}

// ---------------------------------------------------------------- tracker
void Tracker::predict(Track& t, double dt) const {
    cv::Matx44d F = cv::Matx44d::eye();
    F(0, 2) = F(1, 3) = dt;
    t.x = F * t.x;
    cv::Matx44d Q = cv::Matx44d::diag(cv::Vec4d(.05, .05, .5, .5)) * dt;
    t.P = F * t.P * F.t() + Q;
    t.miss += 1;
}

void Tracker::update(Track& t, const Fused& d) const {
    cv::Matx22d S(t.P(0, 0) + 0.3, t.P(0, 1), t.P(1, 0), t.P(1, 1) + 0.3);
    cv::Matx<double, 4, 2> Pc;
    for (int r = 0; r < 4; ++r) { Pc(r, 0) = t.P(r, 0); Pc(r, 1) = t.P(r, 1); }
    cv::Matx<double, 4, 2> K = Pc * S.inv();
    const cv::Vec2d innov(d.xy.x - t.x(0), d.xy.y - t.x(1));
    t.x = t.x + K * innov;
    cv::Matx44d KH = cv::Matx44d::zeros();
    for (int r = 0; r < 4; ++r) { KH(r, 0) = K(r, 0); KH(r, 1) = K(r, 1); }
    t.P = (cv::Matx44d::eye() - KH) * t.P;
    const cv::Point2d c = w_.clamp(t.xy());  // never leave the ground box
    t.x(0) = c.x; t.x(1) = c.y;
    for (size_t i = 0; i < t.emb.size(); ++i) t.emb[i] = 0.9f * t.emb[i] + 0.1f * d.emb[i];
    normalize(t.emb);
    t.miss = 0;
    t.hits += 1;
    t.cams = d.cams;
    t.trail.push_back(t.xy());
    if (t.trail.size() > 200) t.trail.erase(t.trail.begin(), t.trail.begin() + 100);
}

std::vector<TrackView> Tracker::step(const std::vector<Fused>& D, double dt) {
    // drop stale tracks, predict the rest
    tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
                                 [&](const Track& t) { return t.miss * dt > w_.P.maxMissS; }),
                  tracks_.end());
    for (auto& t : tracks_) {
        predict(t, dt);
        const cv::Point2d c = w_.clamp(t.xy());
        t.x(0) = c.x; t.x(1) = c.y;
    }

    const size_t nT = tracks_.size();  // new tracks are appended below, matching only uses the old ones
    std::vector<int> det2trk(D.size(), -1);
    if (nT && !D.empty()) {
        std::vector<std::vector<double>> cost(nT, std::vector<double>(D.size(), 1e3));
        for (size_t i = 0; i < nT; ++i) {
            const double gate = w_.P.gate + 1.0 * tracks_[i].miss * dt;  // lost tracks may have moved further
            for (size_t j = 0; j < D.size(); ++j) {
                const double d = cv::norm(tracks_[i].xy() - D[j].xy) / gate;
                if (d < 1) cost[i][j] = d + 0.5 * (1 - dot(tracks_[i].emb, D[j].emb));
            }
        }
        const std::vector<int> asg = hungarian(cost);
        for (size_t i = 0; i < nT; ++i)
            if (asg[i] >= 0 && cost[i][asg[i]] < 1e3) det2trk[asg[i]] = static_cast<int>(i);
    }

    for (size_t j = 0; j < D.size(); ++j) {
        if (det2trk[j] >= 0) {
            update(tracks_[det2trk[j]], D[j]);
        } else if (static_cast<int>(tracks_.size()) < w_.P.people) {
            Track t;
            t.id = nextId_++;
            t.x = cv::Matx41d(D[j].xy.x, D[j].xy.y, 0, 0);
            t.P = cv::Matx44d::diag(cv::Vec4d(1, 1, 4, 4));
            t.emb = D[j].emb;
            t.cams = D[j].cams;
            t.trail.push_back(D[j].xy);
            tracks_.push_back(std::move(t));
        }
    }

    std::vector<TrackView> out;
    for (auto& t : tracks_) {
        t.confirmed = t.confirmed || t.hits >= w_.P.confirm || t.cams.size() >= 2;
        if (t.miss == 0 && t.confirmed) {
            TrackView v;
            v.id = t.id;
            v.xy = t.xy();
            const size_t n = std::min<size_t>(t.trail.size(), 40);
            v.trail.assign(t.trail.end() - n, t.trail.end());
            out.push_back(std::move(v));
        }
    }
    return out;
}
