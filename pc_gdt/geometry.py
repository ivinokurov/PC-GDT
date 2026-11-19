"""Scene geometry of the PC-GDT digital twin.

Conventions: world axes (east, north, up).  Azimuths are GEOGRAPHIC (deg,
clockwise from North); the paper's scene azimuth theta = 90 deg - theta_geo,
which makes sun_direction() identical to eq. (1) of the paper.
Plane of a face: n . x = c  (paper's d_m = -c).
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


def sun_direction(azimuth_deg, elevation_deg) -> np.ndarray:
    """Unit solar-ray propagation vector (..., 3), components (east, north, up).
    Horizontal part is anti-solar, vertical component -sin(el) < 0."""
    az = np.deg2rad(np.asarray(azimuth_deg, dtype=float))
    el = np.deg2rad(np.asarray(elevation_deg, dtype=float))
    dx = -np.cos(el) * np.sin(az)
    dy = -np.cos(el) * np.cos(az)
    dz = -np.sin(el)
    return np.stack(np.broadcast_arrays(dx, dy, dz), axis=-1)


def antisolar_azimuth(azimuth_deg):
    return (np.asarray(azimuth_deg, dtype=float) + 180.0) % 360.0


def angular_difference_deg(a, b):
    """Signed smallest difference, deg in [-180, 180)."""
    d = (np.asarray(a, float) - np.asarray(b, float)) % 360.0
    return (d + 180.0) % 360.0 - 180.0


def sun_embedding(azimuth_deg, elevation_deg) -> np.ndarray:
    """[cos az, sin az, cos el, sin el], shape (N, 4)  (paper eq. (3))."""
    az = np.deg2rad(np.atleast_1d(np.asarray(azimuth_deg, float)))
    el = np.deg2rad(np.atleast_1d(np.asarray(elevation_deg, float)))
    return np.stack([np.cos(az), np.sin(az), np.cos(el), np.sin(el)], axis=-1)


@dataclass
class RoofFace:
    vertices: np.ndarray            # (M, 3) convex polygon, world coords
    kind: str = "pitched"           # "pitched" | "flat"

    def normal(self) -> np.ndarray:
        v = self.vertices
        n = np.cross(v[1] - v[0], v[-1] - v[0])
        n = n / (np.linalg.norm(n) + 1e-12)
        return n if n[2] >= 0 else -n

    def plane_offset(self) -> float:
        return float(self.normal() @ self.vertices[0])

    def contains_xy(self, p_xy, tol: float = 1e-6) -> np.ndarray:
        """Orientation-robust convex point-in-polygon (boundary included)."""
        v = self.vertices[:, :2]
        pts = np.atleast_2d(p_xy)
        area2 = float(np.sum(v[:, 0] * np.roll(v[:, 1], -1)
                             - np.roll(v[:, 0], -1) * v[:, 1]))
        sgn = 1.0 if area2 >= 0 else -1.0
        ok = np.ones(len(pts), dtype=bool)
        for i in range(len(v)):
            a, b = v[i], v[(i + 1) % len(v)]
            e, r = b - a, pts - a
            ok &= sgn * (e[0] * r[:, 1] - e[1] * r[:, 0]) >= -tol
        return ok


@dataclass
class Canopy:
    centre: np.ndarray              # (3,)
    radius: float

    def sample_points(self, n: int = 64, rng=None) -> np.ndarray:
        rng = rng or np.random.default_rng(0)
        u = rng.normal(size=(n, 3))
        u /= np.linalg.norm(u, axis=1, keepdims=True) + 1e-12
        return self.centre[None, :] + self.radius * u


@dataclass
class Scene:
    faces: list[RoofFace]
    canopies: list[Canopy]
    sun_azimuth: float
    sun_elevation: float
    ridge_segments: list = field(default_factory=list)

    @property
    def sun_vector(self) -> np.ndarray:
        return sun_direction(self.sun_azimuth, self.sun_elevation)


def project_point_on_face(point, face: RoofFace, d, eps: float = 1e-9):
    """Ray-plane intersection (paper eq. (2)); None if unlit / t<=0 / outside."""
    n = face.normal()
    denom = float(n @ d)
    if denom >= 0:
        return None
    t = (face.plane_offset() - float(n @ point)) / denom
    if t <= eps:
        return None
    hit = point + t * d
    if not bool(face.contains_xy(hit[:2])[0]):
        return None
    return t, hit


def shadow_points_on_face(canopy: Canopy, face: RoofFace, d,
                          n_samples: int = 256, occluders=None, rng=None) -> np.ndarray:
    """Vectorised shadow casting.  Returns (K, 2) hit (x, y) surviving:
    positive ray parameter, hit inside the face, and no occlusion of the
    SUN->POINT ray (direction -d) by other canopies, as in the paper."""
    rng = rng or np.random.default_rng(0)
    n = face.normal()
    denom = float(n @ d)
    if denom >= 0:
        return np.zeros((0, 2))
    pts = canopy.sample_points(n_samples, rng)
    t = (face.plane_offset() - pts @ n) / denom
    keep = t > 1e-9
    pts, t = pts[keep], t[keep]
    if not len(pts):
        return np.zeros((0, 2))
    hits = pts + t[:, None] * d
    keep = face.contains_xy(hits[:, :2])
    pts, hits = pts[keep], hits[keep]
    if occluders:
        u = -d                                   # towards the sun
        blocked = np.zeros(len(pts), dtype=bool)
        for o in occluders:
            if o is canopy:
                continue
            w = o.centre - pts
            tc = w @ u
            dist2 = np.einsum("ij,ij->i", w, w) - tc ** 2
            blocked |= (tc > 1e-9) & (dist2 < o.radius ** 2)
        hits = hits[~blocked]
    return hits[:, :2]


def shadow_of_canopy_on_face(canopy, face, d, n_samples=256, occluders=None, rng=None):
    """Back-compatible wrapper."""
    return shadow_points_on_face(canopy, face, d, n_samples, occluders, rng)


def _shared_ridge(scene: Scene, fa: RoofFace, fb: RoofFace, tol=1e-3):
    for r0, r1 in scene.ridge_segments:
        ok = True
        for r in (r0, r1):
            for f in (fa, fb):
                if abs(float(f.normal() @ r) - f.plane_offset()) > tol:
                    ok = False
        if ok:
            return r0, r1
    return None


def ridge_break_of_shadow(scene, face_idx_a, face_idx_b, canopy,
                          n_samples=512, rng=None, break_tol=1.5):
    """Estimate of the shadow-contour kink on the ridge shared by two slopes:
    the ridge point where the shadow approaches from BOTH sides.
    Returns (2,) world (x, y) or None.  Approximate (sampling-based)."""
    rng = rng or np.random.default_rng(0)
    fa, fb = scene.faces[face_idx_a], scene.faces[face_idx_b]
    d = scene.sun_vector
    pa = shadow_points_on_face(canopy, fa, d, n_samples, scene.canopies, rng)
    pb = shadow_points_on_face(canopy, fb, d, n_samples, scene.canopies, rng)
    if not len(pa) or not len(pb):
        return None
    ridge = _shared_ridge(scene, fa, fb)
    if ridge is None:
        return None
    r0, r1 = ridge
    s = np.linspace(0, 1, 128)[:, None]
    rp = r0[None, :2] * (1 - s) + r1[None, :2] * s
    da = np.min(np.linalg.norm(rp[:, None] - pa[None], axis=-1), axis=1)
    db = np.min(np.linalg.norm(rp[:, None] - pb[None], axis=-1), axis=1)
    score = np.maximum(da, db)
    k = int(np.argmin(score))
    return rp[k] if score[k] < break_tol else None


def make_gabled_house(cx, cy, w=12.0, l=18.0, h_ridge=4.0, eave_h=2.5,
                      ridge_along_y=True):
    if ridge_along_y:
        hw, hl = w / 2, l / 2
        left = np.array([[-hw, -hl, eave_h], [-hw, hl, eave_h],
                         [0, hl, h_ridge], [0, -hl, h_ridge]])
        right = np.array([[hw, -hl, eave_h], [hw, hl, eave_h],
                          [0, hl, h_ridge], [0, -hl, h_ridge]])
        ridge = (np.array([0, -hl, h_ridge]), np.array([0, hl, h_ridge]))
    else:
        hw, hl = l / 2, w / 2
        left = np.array([[-hw, -hl, eave_h], [hw, -hl, eave_h],
                         [hw, 0, h_ridge], [-hw, 0, h_ridge]])
        right = np.array([[-hw, hl, eave_h], [hw, hl, eave_h],
                          [hw, 0, h_ridge], [-hw, 0, h_ridge]])
        ridge = (np.array([-hw, 0, h_ridge]), np.array([hw, 0, h_ridge]))
    off = np.array([cx, cy, 0.0])
    return [RoofFace(left + off), RoofFace(right + off)], tuple(r + off for r in ridge)


def make_flat_house(cx, cy, w=10.0, l=14.0, h=3.0):
    hw, hl = w / 2, l / 2
    v = np.array([[-hw, -hl, h], [-hw, hl, h], [hw, hl, h], [hw, -hl, h]])
    return [RoofFace(v + [cx, cy, 0], kind="flat")]


def illuminated_faces(scene: Scene) -> list[int]:
    d = scene.sun_vector
    return [i for i, f in enumerate(scene.faces) if float(f.normal() @ d) < 0.0]


def random_scenes(n=64, seed=0) -> list[Scene]:
    """Random cottage scenes; canopies back-projected from an illuminated
    slope so the anti-solar footprint lands on the roof.  Canopy centres are
    kept above ridge + radius (no roof intersection)."""
    rng = np.random.default_rng(seed)
    scenes = []
    for _ in range(n):
        faces, ridge = make_gabled_house(0.0, 0.0)
        if rng.random() < 0.25:
            f2 = make_flat_house(35.0, 10.0)
            ridges = [ridge]
        else:
            f2, r2 = make_gabled_house(35.0, 10.0, ridge_along_y=False)
            ridges = [ridge, r2]
        all_faces = faces + f2
        az, el = float(rng.uniform(0, 360)), float(rng.uniform(15, 60))
        d = sun_direction(az, el)
        canopies = []
        for _ in range(int(rng.integers(2, 5))):
            idx = int(rng.choice(illuminated_faces(Scene(all_faces, [], az, el))))
            v = all_faces[idx].vertices
            tgt = (v[0] + v[2]) / 2
            tgt = tgt + rng.uniform(-.15, .15) * (v[1] - v[0]) \
                      + rng.uniform(-.15, .15) * (v[3] - v[0])
            r = float(rng.uniform(2.0, 3.5))
            zc = float(rng.uniform(max(5.0, 4.0 + r + .5), max(8.0, 4.0 + r + 1.5)))
            s = (zc - tgt[2]) / max(-d[2], 1e-6)
            canopies.append(Canopy(tgt + (-d) * s, r))
        scenes.append(Scene(all_faces, canopies, az, el, ridges))
    return scenes


if __name__ == "__main__":
    sc = random_scenes(4)[0]
    d = sc.sun_vector
    assert abs(np.linalg.norm(d) - 1) < 1e-9 and d[2] < 0
    tot = sum(len(shadow_points_on_face(sc.canopies[0], sc.faces[fi], d,
                                        occluders=sc.canopies))
              for fi in illuminated_faces(sc))
    print("az/el:", round(sc.sun_azimuth, 2), round(sc.sun_elevation, 2),
          "| shadow samples:", tot, "| emb:", sun_embedding(sc.sun_azimuth, sc.sun_elevation)[0])