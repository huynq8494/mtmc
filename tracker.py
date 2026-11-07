import numpy as np
from scipy.optimize import linear_sum_assignment as hungarian
import config as C


def fuse(dets):
    """dets: list of (xy, emb, cam, range). Greedy cluster (closest cameras first, one det per camera
    per cluster) -> list of (xy, emb, cams)."""
    cl = []
    for xy, e, cam, r in sorted(dets, key=lambda d: d[3]):
        thr = C.FUSE_D + 0.03 * r
        best, bd = None, thr
        for c in cl:
            d = np.linalg.norm(c["xy"] - xy)
            if d < bd and cam not in c["cams"]:
                best, bd = c, d
        if best is None:
            cl.append(dict(xy=xy, w=1 / r**2, ps=[(xy, 1 / r**2)], es=[e], cams=[cam]))
        else:
            best["ps"].append((xy, 1 / r**2)); best["es"].append(e); best["cams"].append(cam)
            w = np.array([p[1] for p in best["ps"]])
            best["xy"] = (np.array([p[0] for p in best["ps"]]) * w[:, None]).sum(0) / w.sum()
    out = []
    for c in cl:
        e = np.mean(c["es"], 0)
        out.append((c["xy"], e / np.linalg.norm(e), c["cams"]))
    return out


class Track:
    def __init__(self, id, xy, emb):
        self.id, self.x, self.P = id, np.r_[xy, 0, 0], np.diag([1, 1, 4, 4.])
        self.emb, self.miss, self.hits, self.cams, self.confirmed = emb, 0, 1, [], False
        self.trail = [tuple(xy)]

    def predict(self, dt):
        F = np.eye(4); F[0, 2] = F[1, 3] = dt
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + np.diag([.05, .05, .5, .5]) * dt
        self.miss += 1

    def update(self, xy, emb, cams):
        S = self.P[:2, :2] + np.eye(2) * 0.3
        K = self.P[:, :2] @ np.linalg.inv(S)
        self.x = self.x + K @ (xy - self.x[:2])
        self.P = (np.eye(4) - np.c_[K, np.zeros((4, 2))]) @ self.P
        self.x[:2] = C.clamp(self.x[:2])  # never leave the ground box
        self.emb = 0.9 * self.emb + 0.1 * emb
        self.emb /= np.linalg.norm(self.emb)
        self.miss, self.hits, self.cams = 0, self.hits + 1, cams
        self.trail.append(tuple(self.x[:2]))

    @property
    def xy(self):
        return self.x[:2]


class Tracker:
    def __init__(self):
        self.tracks, self.next_id = [], 1

    def step(self, fused, dt):
        self.tracks = [t for t in self.tracks if t.miss * dt <= C.MAX_MISS_S]  # drop stale tracks
        for t in self.tracks:
            t.predict(dt)
            t.x[:2] = C.clamp(t.x[:2])
        T, D = self.tracks, fused
        if T and D:
            cost = np.zeros((len(T), len(D)))
            for i, t in enumerate(T):
                gate = C.GATE + 1.0 * t.miss * dt   # lost tracks may have moved further
                for j, (xy, e, _) in enumerate(D):
                    d = np.linalg.norm(t.xy - xy) / gate
                    cost[i, j] = d + 0.5 * (1 - t.emb @ e) if d < 1 else 1e3
            r, c = hungarian(cost)
            m = {j: i for i, j in zip(r, c) if cost[i, j] < 1e3}
        else:
            m = {}
        for j, (xy, e, cams) in enumerate(D):
            if j in m:
                T[m[j]].update(xy, e, cams)
            elif len(T) < C.PEOPLE:
                T.append(Track(self.next_id, xy, e)); self.next_id += 1
                T[-1].cams = cams
        for t in T:
            t.confirmed = t.confirmed or t.hits >= C.CONFIRM or len(t.cams) >= 2
        return [t for t in T if t.miss == 0 and t.confirmed]
