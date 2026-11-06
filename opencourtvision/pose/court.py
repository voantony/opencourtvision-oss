"""Image -> court-plane homography so distances come out in centimetres "from the court's known size".

Give the four doubles-court corners in the image (near-left, near-right, far-right, far-left, as seen from the camera).
Court coords: x across the court (0 = near-left doubles sideline), y along it (0 = near baseline), metres.
"""
import numpy as np

COURT_W, COURT_L = 10.97, 23.77      # doubles width, length (m); padel would be 10 x 20


def homography(src, dst):
    """3x3 H with dst ~ H @ src (DLT on >= 4 point pairs)."""
    src = np.asarray(src, float); dst = np.asarray(dst, float); A = []
    for (x, y), (u, v) in zip(src, dst):
        A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    H = np.linalg.svd(np.asarray(A))[2][-1].reshape(3, 3)
    return H / H[2, 2]


def apply(H, pts):
    pts = np.asarray(pts, float); p = np.c_[pts.reshape(-1, 2), np.ones(pts.size // 2)] @ H.T
    return (p[:, :2] / p[:, 2:]).reshape(pts.shape)


class Court:
    def __init__(self, corners_px, width=COURT_W, length=COURT_L):
        self.width, self.length = width, length
        self.H = homography(corners_px, [(0, 0), (width, 0), (width, length), (0, length)])
        self.Hinv = np.linalg.inv(self.H)

    @classmethod
    def parse(cls, s, **kw):
        """'x1,y1,x2,y2,x3,y3,x4,y4' (near-left, near-right, far-right, far-left)."""
        v = [float(t) for t in s.split(",")]
        if len(v) != 8: raise ValueError("--court needs 8 numbers: x,y for near-left, near-right, far-right, far-left")
        return cls(np.reshape(v, (4, 2)), **kw)

    def to_court(self, px):
        """Image px (..., 2) on the ground plane -> court metres (..., 2). NaN in, NaN out."""
        return apply(self.H, px)

    def cm_per_px(self, px):
        """Local scale (cm per image px) at ground points (T, 2), from a 1 px horizontal step. Used to turn
        body-plane pixel distances into cm (approximate: assumes the body plane faces the camera)."""
        px = np.asarray(px, float)
        d = self.to_court(px + [1.0, 0.0]) - self.to_court(px)
        return np.hypot(d[..., 0], d[..., 1]) * 100.0

    def baseline_angle(self, a_px, b_px):
        """Angle (deg, 0-90) between the court line a->b (e.g. back foot -> front foot) and the baseline."""
        d = self.to_court(b_px) - self.to_court(a_px)
        ang = np.degrees(np.arctan2(np.abs(d[..., 1]), np.abs(d[..., 0])))
        return ang
