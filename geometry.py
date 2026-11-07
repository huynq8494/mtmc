import numpy as np
import config as C


class Camera:
    def __init__(self, i):
        """Pose from the parsed camera config. Camera looks along local -Z, +Y is up."""
        self.id = i
        self.pos = np.array(C.CAMERAS[i]["position"], float)
        self.R = np.array(C.CAMERAS[i]["rotation"], float)  # camera -> world
        self.fwd = self.R @ [0, 0, -1]
        self.up = self.R @ [0, 1, 0]

    def ground(self, uv):
        """Pixels (N,2) -> ground (z=0) world xy (N,2), valid mask, range (m)."""
        uv = np.asarray(uv, float).reshape(-1, 2)
        d = np.stack([(uv[:, 0] - C.W / 2) / C.F, -(uv[:, 1] - C.H / 2) / C.F, -np.ones(len(uv))], 1) @ self.R.T
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        ok = d[:, 2] < -1e-3
        t = np.where(ok, -self.pos[2] / np.where(ok, d[:, 2], -1), 0)
        ok &= t < C.MAX_RANGE
        return self.pos[:2] + t[:, None] * d[:, :2], ok, t

    def project(self, xyz):
        """World points (N,3) -> pixels (N,2), in-image mask."""
        p = (np.asarray(xyz, float) - self.pos) @ self.R  # camera frame
        z = -p[:, 2]
        z_ = np.where(z > 1e-3, z, 1)
        uv = np.stack([p[:, 0] / z_ * C.F + C.W / 2, -p[:, 1] / z_ * C.F + C.H / 2], 1)
        return uv, (z > 1e-3) & (uv[:, 0] >= 0) & (uv[:, 0] < C.W) & (uv[:, 1] >= 0) & (uv[:, 1] < C.H)


def check(ids):
    """Print camera pose sanity: direction, roll, nearest visible ground point and floor coverage."""
    gx, gy = np.meshgrid(np.arange(C.XMIN, C.XMAX + .1, 1.0), np.arange(C.YMIN, C.YMAX + .1, 1.0))
    grid = np.c_[gx.ravel(), gy.ravel(), np.zeros(gx.size)]
    cover = np.zeros(len(grid), int)
    print(f"bounds x[{C.XMIN},{C.XMAX}] y[{C.YMIN},{C.YMAX}]  HFOV={C.HFOV:.1f}deg F={C.F:.0f}px")
    for i in ids:
        c = Camera(i)
        vis = c.project(grid)[1]
        cover += vis
        xy, ok, r = c.ground([[C.W / 2, C.H - 1], [0, C.H - 1], [C.W - 1, C.H - 1]])
        flags = []
        if c.up[2] < 0: flags.append("UPSIDE-DOWN")
        if not ok[0]: flags.append("NO-GROUND-AT-BOTTOM")
        elif not C.inside(xy[0], 2): flags.append("LOOKS-OUTSIDE-BOUNDS")
        if vis.mean() < 0.05: flags.append("SEES-<5%-OF-FLOOR")
        print(f"cam{i:02d} pos={c.pos} fwd={np.round(c.fwd, 2)} up_z={c.up[2]:+.2f} "
              f"bottom_hit={np.round(xy[0], 1) if ok[0] else None} floor_seen={vis.mean():.0%} {' '.join(flags)}")
    print(f"floor covered by >=1 cam: {(cover >= 1).mean():.0%}, >=2 cams: {(cover >= 2).mean():.0%}")
