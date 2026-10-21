#!/usr/bin/env python3
"""Identify stars in a (simulated) star-field image and estimate the camera attitude.

Lost-in-space solver:
  1. detect stars (local background subtraction, smoothed threshold, flux-weighted centroids)
  2. turn centroids into unit rays with a pinhole model
  3. for pairs of bright detections, look up catalogue pairs with the same angular separation,
     build an attitude for every candidate (TRIAD) and verify it against ALL detections
     using a KD-tree on the catalogue
  4. refine with a least-squares (Wahba/SVD) fit, optionally refine the focal length / FOV

Both image orientations are tried automatically:
  * "normal"   - image as a real camera sees the sky (north up => east is LEFT)
  * "mirrored" - east to the RIGHT (common in naive simulators / sky-chart style renders)

Catalogue: whitespace or comma separated text. Default column order is RA_deg Dec_deg ID mag;
use --cols to change it (e.g. --cols id,ra,dec,mag) and --ra-hours if RA is in hours.

Install: pip install numpy scipy pillow
Example: python starfield_boresight.py --image simulated.png --catalog scHIP4to6p5.txt --fov 15 \
             --out results/annotated.png --csv results/matches.csv
"""
import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.spatial import cKDTree
from scipy.stats import binom


# ----------------------------------------------------------------------------- geometry helpers
def chord(deg):
    """Chord length on the unit sphere for an angle in degrees."""
    return 2.0 * math.sin(math.radians(deg) / 2.0)


def chord_to_deg(c):
    return np.degrees(2.0 * np.arcsin(np.clip(np.asarray(c) / 2.0, 0.0, 1.0)))


def ang_deg(u, v):
    return float(chord_to_deg(np.linalg.norm(u - v)))


def radec_to_vec(ra_deg, dec_deg):
    r, d = np.radians(ra_deg), np.radians(dec_deg)
    return np.c_[np.cos(d) * np.cos(r), np.cos(d) * np.sin(r), np.sin(d)]


def triad(a, b):
    """Orthonormal frames (columns) from direction a and the plane of (a, b). a, b: (M,3) unit vectors."""
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)
    t2 = np.cross(a, b)
    t2 /= np.linalg.norm(t2, axis=1, keepdims=True)
    t3 = np.cross(a, t2)
    return np.stack([a, t2, t3], axis=2)


def wahba(inertial, camera):
    """Rotation R minimising sum |camera_i - R inertial_i|^2 (SVD solution)."""
    B = camera.T @ inertial
    U, _, Vt = np.linalg.svd(B)
    D = np.eye(3)
    D[2, 2] = np.linalg.det(U @ Vt)
    return U @ D @ Vt


# ----------------------------------------------------------------------------- catalogue
def load_cat(path, cols='ra,dec,id,mag', maxmag=None, ra_hours=False):
    names = [c.strip().lower() for c in cols.split(',')]
    if 'ra' not in names or 'dec' not in names:
        raise ValueError("--cols must contain 'ra' and 'dec'")
    col = {n: i for i, n in enumerate(names) if n in ('ra', 'dec', 'id', 'mag')}
    need = max(col.values()) + 1
    ra, de, ids, mags = [], [], [], []
    skipped = 0
    with open(path, encoding='utf-8-sig', errors='replace') as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith(('#', '%', ';', '//')):
                continue
            p = next(csv.reader([s])) if ',' in s else s.split()
            p = [x.strip() for x in p]
            if len(p) < need:
                skipped += 1
                continue
            try:
                r = float(p[col['ra']])
                d = float(p[col['dec']])
                m = float(p[col['mag']]) if 'mag' in col else 0.0
            except ValueError:
                skipped += 1          # header line or junk
                continue
            if ra_hours:
                r *= 15.0
            if r == 360.0:
                r = 0.0
            if not (0 <= r < 360 and -90 <= d <= 90 and np.isfinite(m)):
                skipped += 1
                continue
            if maxmag is not None and m > maxmag:
                continue
            if 'id' in col:
                sid = p[col['id']]
                try:
                    fv = float(sid)
                    sid = str(int(fv)) if fv.is_integer() else sid
                except ValueError:
                    pass
            else:
                sid = str(len(ids))
            ra.append(r); de.append(d); ids.append(sid); mags.append(m)
    if not ra:
        raise ValueError(f'No usable catalogue rows (skipped {skipped}). Check --cols ({cols}).')
    ra, de, mags = np.array(ra), np.array(de), np.array(mags)
    warn = []
    if skipped > 0.2 * (len(ra) + skipped):
        warn.append(f'{skipped} catalogue lines were skipped - is the column order (--cols {cols}) right?')
    if not ra_hours and len(ra) > 50 and ra.max() <= 24.0:
        warn.append('All RA values are <= 24: RA may be in HOURS - try --ra-hours.')
    if len(ra) > 50 and np.ptp(mags) == 0 and 'mag' in col:
        warn.append('All magnitudes equal - check that the mag column is correct.')
    return ra, de, ids, mags, radec_to_vec(ra, de), warn


# ----------------------------------------------------------------------------- image / detection
def read_gray(path):
    im = Image.open(path)
    if im.mode in ('I', 'I;16', 'I;16B', 'I;16L', 'F', 'L'):
        return np.asarray(im, dtype=float)
    a = np.asarray(im.convert('RGB'), dtype=float)
    return 0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]


def estimate_background(img, box=32):
    """Median background on a coarse grid (window ~ box px, much larger than a star), upsampled."""
    s = max(1, box // 8)
    small = img[::s, ::s]
    bgs = ndimage.median_filter(small, size=9, mode='reflect')
    bg = np.repeat(np.repeat(bgs, s, axis=0), s, axis=1)[:img.shape[0], :img.shape[1]]
    return ndimage.uniform_filter(bg, size=s) if s > 1 else bg


def detect(img, sigma=4.0, maxstars=80, minarea=1, maxarea=1000, bg_box=32, border=2):
    res = img - estimate_background(img, bg_box)
    sm = ndimage.gaussian_filter(res, 1.0)              # matched-ish filter to suppress noise
    med = float(np.median(sm))
    mad = float(np.median(np.abs(sm - med)))
    noise = max(1.4826 * mad, 1e-4 * float(sm.max() - med), 1e-12)   # floor for noiseless renders
    thr = med + sigma * noise
    mask = sm > thr
    if border > 0:
        mask[:border, :] = mask[-border:, :] = False
        mask[:, :border] = mask[:, -border:] = False
    labels, n = ndimage.label(mask)
    if n == 0:
        return [], thr, noise
    idx = np.arange(1, n + 1)
    w = np.clip(res - float(np.median(res)), 0, None)
    area = ndimage.sum(mask, labels, idx)
    flux = ndimage.sum(w, labels, idx)
    com = ndimage.center_of_mass(w, labels, idx)
    out = []
    for a, f, (cy, cx) in zip(area, flux, com):
        if minarea <= a <= maxarea and f > 0 and np.isfinite(cx):
            out.append({'x': float(cx), 'y': float(cy), 'flux': float(f), 'area': int(a)})
    out.sort(key=lambda d: d['flux'], reverse=True)
    return out[:maxstars], thr, noise


# ----------------------------------------------------------------------------- camera model
def focal_px(w, h, fovx, fovy):
    return (w / 2) / math.tan(math.radians(fovx) / 2), (h / 2) / math.tan(math.radians(fovy) / 2)


def pixel_rays(xy, w, h, fxp, fyp, parity=1):
    """Camera frame: x = image right (negated if parity=-1), y = image DOWN, z = boresight.
    This is a right-handed frame, i.e. it matches a real camera when parity=+1."""
    cx, cy = (w - 1) / 2, (h - 1) / 2
    v = np.c_[parity * (xy[:, 0] - cx) / fxp, (xy[:, 1] - cy) / fyp, np.ones(len(xy))]
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def project(R, vec, w, h, fxp, fyp, parity=1):
    b = vec @ R.T
    ok = b[:, 2] > 1e-9
    z = np.where(ok, b[:, 2], 1.0)
    u = (w - 1) / 2 + parity * fxp * b[:, 0] / z
    v = (h - 1) / 2 + fyp * b[:, 1] / z
    return u, v, ok


# ----------------------------------------------------------------------------- solver
class Solver:
    def __init__(self, cat, max_pair_deg, pair_tol, match_tol, min_matches, chunk=4000):
        self.cat = cat
        self.tree = cKDTree(cat)
        self.pair_tol = pair_tol
        self.match_tol = match_tol
        self.ctol = chord(match_tol)
        self.min_matches = min_matches
        self.max_pair = max_pair_deg
        self.min_sep = max(0.2, 5 * pair_tol)
        self.chunk = chunk
        # probability that a random direction lands within match_tol of some catalogue star
        # (x3 safety factor for density variations, e.g. the galactic plane)
        self.p_rand = min(0.5, 3 * len(cat) * math.radians(match_tol) ** 2 / 4)
        pairs = self.tree.query_pairs(chord(max_pair_deg), output_type='ndarray')
        if not len(pairs):
            raise ValueError('No catalogue pairs fit inside the field of view; check catalogue and FOV.')
        ang = chord_to_deg(np.linalg.norm(cat[pairs[:, 0]] - cat[pairs[:, 1]], axis=1))
        order = np.argsort(ang)
        self.pairs, self.ang = pairs[order], ang[order]

    def match(self, R, obs):
        """Unique nearest-neighbour matching of every detection to the catalogue."""
        inert = obs @ R                                   # rows = R.T @ obs_i
        k = min(3, len(self.cat))
        d, idx = self.tree.query(inert, k=k, distance_upper_bound=self.ctol)
        if k == 1:
            d, idx = d[:, None], idx[:, None]
        o, kk = np.nonzero(np.isfinite(d))
        used_o, used_c, out = set(), set(), []
        for t in np.argsort(d[o, kk]):
            oi, ci = int(o[t]), int(idx[o[t], kk[t]])
            if oi not in used_o and ci not in used_c:
                used_o.add(oi); used_c.add(ci)
                out.append((oi, ci, float(chord_to_deg(d[o[t], kk[t]]))))
        return out

    def false_alarm(self, k, n, tested):
        """Expected number of random hypotheses (out of `tested`) reaching k matches by chance.
        Two matches are built into every hypothesis; the rest are binomial(n-2, p_rand)."""
        return float(binom.sf(k - 3, max(n - 2, 1), self.p_rand)) * max(tested, 1) * 3

    @staticmethod
    def rms(m):
        return float(np.sqrt(np.mean([x[2] ** 2 for x in m]))) if m else float('inf')

    def refine(self, R, obs, iters=6):
        m = self.match(R, obs)
        for _ in range(iters):
            if len(m) < 3:
                break
            R2 = wahba(self.cat[[x[1] for x in m]], obs[[x[0] for x in m]])
            m2 = self.match(R2, obs)
            if len(m2) < len(m) or (len(m2) == len(m) and self.rms(m2) >= self.rms(m) - 1e-12):
                if len(m2) >= len(m):
                    R, m = R2, m2
                break
            R, m = R2, m2
        # reject outliers (e.g. stars clipped by the image edge, blended pairs) and refit
        if len(m) >= 6:
            med = float(np.median([x[2] for x in m]))
            keep = [x for x in m if x[2] <= max(5 * med, 0.1 * self.match_tol)]
            if len(keep) >= 4 and len(keep) < len(m):
                R = wahba(self.cat[[x[1] for x in keep]], obs[[x[0] for x in keep]])
                m = [x for x in self.match(R, obs) if x[2] <= max(5 * med, 0.1 * self.match_tol)]
        return R, m, self.rms(m)

    def solve(self, obs_sets, n_pair_stars=12, n_verify=30, max_hyp=3_000_000):
        """obs_sets: list of (parity, rays). Returns (nmatch, rms, parity, R, matches) or None."""
        n = len(obs_sets[0][1])
        nps = min(n_pair_stars, n)
        nv = min(max(n_verify, nps), n)
        need = min(self.min_matches, n)
        strong = min(n, max(self.min_matches + 3, math.ceil(0.4 * nv)))
        pair_order = sorted((i + j, i, j) for i in range(nps) for j in range(i + 1, nps))
        best, tested = None, 0
        for _, i, j in pair_order:
            for parity, obs in obs_sets:
                sep = ang_deg(obs[i], obs[j])
                if sep < self.min_sep or sep > self.max_pair:
                    continue
                lo = np.searchsorted(self.ang, sep - self.pair_tol)
                hi = np.searchsorted(self.ang, sep + self.pair_tol, side='right')
                if hi <= lo:
                    continue
                S = triad(obs[i], obs[j])[0]
                P = obs[:nv] @ S                               # detections expressed in the obs triad
                for s in range(lo, hi, self.chunk):
                    c = self.pairs[s:min(hi, s + self.chunk)]
                    a = np.concatenate([c[:, 0], c[:, 1]])     # both star-to-star assignments
                    b = np.concatenate([c[:, 1], c[:, 0]])
                    T = triad(self.cat[a], self.cat[b])
                    inert = np.einsum('mij,nj->mni', T, P).reshape(-1, 3)
                    d, _ = self.tree.query(inert, distance_upper_bound=self.ctol)
                    cnt = np.isfinite(d).reshape(len(a), nv).sum(axis=1)
                    tested += len(a)
                    good = np.flatnonzero(cnt >= need)
                    for g in good[np.argsort(-cnt[good])][:8]:
                        R, m, r = self.refine(S @ T[g].T, obs)
                        if len(m) < need or self.false_alarm(len(m), n, tested) > 0.01:
                            continue                       # not distinguishable from chance
                        if best is None or (len(m), -r) > (best[0], -best[1]):
                            best = (len(m), r, parity, R, m)
                        if best is not None and best[0] >= strong:
                            return best, tested
                    if tested >= max_hyp:
                        return best, tested
        return best, tested


def refine_focal(xy, w, h, fxp, fyp, parity, matches, cat):
    """1-D search for a focal-length scale k minimising matched-star residuals."""
    o = [m[0] for m in matches]
    c = cat[[m[1] for m in matches]]

    def cost(k):
        ob = pixel_rays(xy[o], w, h, fxp * k, fyp * k, parity)
        R = wahba(c, ob)
        return float(np.mean(np.sum((ob - c @ R.T) ** 2, axis=1)))

    ks = np.linspace(0.97, 1.03, 301)
    k0 = ks[int(np.argmin([cost(k) for k in ks]))]
    ks = np.linspace(k0 - 2.5e-4, k0 + 2.5e-4, 51)
    return float(ks[int(np.argmin([cost(k) for k in ks]))])


def fov_factors(pct, step=0.5):
    out = [1.0]
    k = step
    while k <= pct + 1e-9:
        out += [1 - k / 100, 1 + k / 100]
        k += step
    return out


# ----------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--image', required=True)
    p.add_argument('--catalog', required=True)
    p.add_argument('--fov', '--fov-x', dest='fx', type=float, required=True,
                   help='Horizontal (image width) field of view in degrees')
    p.add_argument('--fov-y', type=float,
                   help='Vertical FOV in degrees (default: derived from --fov assuming square pixels)')
    p.add_argument('--cols', default='ra,dec,id,mag',
                   help="Catalogue column order, e.g. 'id,ra,dec,mag' (default: ra,dec,id,mag)")
    p.add_argument('--ra-hours', action='store_true', help='Catalogue RA is in hours')
    p.add_argument('--max-mag', type=float, help='Ignore catalogue stars fainter than this')
    p.add_argument('--threshold-sigma', type=float, default=4)
    p.add_argument('--max-stars', type=int, default=80)
    p.add_argument('--min-area', type=int, default=1)
    p.add_argument('--max-area', type=int, default=1000)
    p.add_argument('--bg-box', type=int, default=32, help='Background window in pixels (>> star size)')
    p.add_argument('--pair-stars', type=int, default=12, help='Brightest detections used to form pairs')
    p.add_argument('--pair-tolerance', type=float, help='Degrees (default: 3 pixels)')
    p.add_argument('--match-tolerance', type=float, help='Degrees (default: 5 pixels)')
    p.add_argument('--min-matches', type=int, default=5)
    p.add_argument('--orientation', choices=['auto', 'normal', 'mirrored'], default='auto')
    p.add_argument('--fov-search', type=float, default=0.0,
                   help='If the FOV is uncertain, also try +/- this many percent (0.5%% steps)')
    p.add_argument('--no-fov-refine', action='store_true')
    p.add_argument('--max-hypotheses', type=int, default=3_000_000)
    p.add_argument('--out', default='results/annotated.png')
    p.add_argument('--csv', help='Optional CSV file for the matched stars')
    a = p.parse_args()
    if not 0 < a.fx < 180:
        p.error('FOV must be between 0 and 180 degrees')

    try:
        img = read_gray(a.image)
        h, w = img.shape
        fovx = a.fx
        fovy = a.fov_y or math.degrees(2 * math.atan(math.tan(math.radians(fovx) / 2) * h / w))
        fxp, fyp = focal_px(w, h, fovx, fovy)
        scale = fovx / w
        pair_tol = a.pair_tolerance or max(0.005, 3 * scale)
        match_tol = a.match_tolerance or max(0.008, 5 * scale)

        ds, thr, noise = detect(img, a.threshold_sigma, a.max_stars, a.min_area, a.max_area, a.bg_box)
        print(f'Image {w}x{h} | FOV {fovx:.4f} x {fovy:.4f} deg | {scale * 3600:.1f} arcsec/px')
        print(f'Detection threshold={thr:.4g}, noise={noise:.4g}; detected {len(ds)} star candidates.')
        if len(ds) < 4:
            raise RuntimeError('Fewer than 4 stars detected - try a lower --threshold-sigma.')

        ra, de, ids, mag, cat, warns = load_cat(a.catalog, a.cols, a.max_mag, a.ra_hours)
        print(f'Loaded {len(ids)} catalogue stars (first: ID {ids[0]} RA={ra[0]:.4f} Dec={de[0]:.4f} mag={mag[0]:.2f}).')
        for wmsg in warns:
            print('WARNING:', wmsg)

        factors = fov_factors(a.fov_search)
        diag = math.degrees(2 * math.atan(math.hypot(math.tan(math.radians(fovx) / 2),
                                                     math.tan(math.radians(fovy) / 2))))
        solver = Solver(cat, diag * max(factors) * 1.02, pair_tol, match_tol, max(4, a.min_matches))
        print(f'{len(solver.pairs)} catalogue pairs; pair tol {pair_tol * 3600:.0f}", match tol {match_tol * 3600:.0f}"')

        parities = {'auto': [1, -1], 'normal': [1], 'mirrored': [-1]}[a.orientation]
        xy = np.array([[d['x'], d['y']] for d in ds])
        result = None
        for k in factors:
            fk = (fxp / k, fyp / k)
            sets = [(par, pixel_rays(xy, w, h, fk[0], fk[1], par)) for par in parities]
            result, tested = solver.solve(sets, a.pair_stars, max_hyp=a.max_hypotheses)
            print(f'  FOV x{k:.3f}: tested {tested} hypotheses -> '
                  + (f'{result[0]} matches' if result else 'no solution'))
            if result:
                fxp, fyp = fk
                break
        if result is None:
            raise RuntimeError('No reliable attitude found. Check FOV (try --fov-search 3), catalogue '
                               'columns (--cols / --ra-hours), magnitude limit and detection settings.')

        nm, rms, parity, R, matches = result
        if not a.no_fov_refine and nm >= 6:
            kf = refine_focal(xy, w, h, fxp, fyp, parity, matches, cat)
            fxp, fyp = fxp * kf, fyp * kf
            obs = pixel_rays(xy, w, h, fxp, fyp, parity)
            R, matches, rms = solver.refine(wahba(cat[[m[1] for m in matches]], obs[[m[0] for m in matches]]), obs)
        fit_fovx = math.degrees(2 * math.atan((w / 2) / fxp))
        fit_fovy = math.degrees(2 * math.atan((h / 2) / fyp))

        # attitude outputs
        bore = R[2]                                     # = R.T @ [0,0,1]
        rb = math.degrees(math.atan2(bore[1], bore[0])) % 360
        db = math.degrees(math.asin(np.clip(bore[2], -1, 1)))
        up = R.T @ np.array([0.0, -1.0, 0.0])           # image "up" direction on the sky
        r0, d0 = math.radians(rb), math.radians(db)
        north = np.array([-math.sin(d0) * math.cos(r0), -math.sin(d0) * math.sin(r0), math.cos(d0)])
        east = np.array([-math.sin(r0), math.cos(r0), 0.0])
        roll = math.degrees(math.atan2(up @ east, up @ north)) % 360

        print(f'\nMatched {len(matches)} of {len(ds)} detections:')
        for oi, ci, r in sorted(matches):
            print(f"  det {oi:02d} pixel=({ds[oi]['x']:8.2f},{ds[oi]['y']:8.2f}) -> ID {ids[ci]:>8}  "
                  f"RA={ra[ci]:11.6f} Dec={de[ci]:+10.6f} mag={mag[ci]:5.2f}  residual={r * 3600:6.2f}\"")
        print(f'\nImage orientation : {"normal (north up => east left, like a real camera)" if parity == 1 else "MIRRORED (east/west flipped)"}')
        print(f'BORESIGHT         : RA={rb:.8f} deg, Dec={db:.8f} deg')
        print(f'Roll (PA of image-up, N->E): {roll:.4f} deg')
        print(f'Fitted FOV        : {fit_fovx:.5f} x {fit_fovy:.5f} deg (given {fovx:.5f})')
        print(f'Matched-star RMS  : {rms * 3600:.2f} arcsec')

        # catalogue stars predicted inside the frame
        near = solver.tree.query_ball_point(bore, chord(diag / 2 * 1.05))
        u, v, ok = project(R, cat[near], w, h, fxp, fyp, parity)
        inside = ok & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        print(f'Catalogue stars predicted inside the frame: {int(inside.sum())}')

        # annotated image (contrast-stretched so 16-bit / dark images are visible)
        lo, hi = np.median(img), np.percentile(img, 99.95)
        disp = (np.clip((img - lo) / max(hi - lo, 1e-9), 0, 1) ** 0.5 * 255).astype(np.uint8)
        im = Image.fromarray(disp).convert('RGB')
        dr = ImageDraw.Draw(im)
        matched_c = {m[1] for m in matches}
        for ci, uu, vv in zip(np.array(near)[inside], u[inside], v[inside]):
            if ci not in matched_c:
                dr.ellipse((uu - 3, vv - 3, uu + 3, vv + 3), outline=(0, 180, 255))
                dr.text((uu + 5, vv + 2), ids[ci], fill=(0, 180, 255))
        by = {m[0]: m for m in matches}
        for i, d in enumerate(ds):
            x, y = d['x'], d['y']
            good = i in by
            dr.ellipse((x - 6, y - 6, x + 6, y + 6), outline=(0, 255, 0) if good else (255, 80, 80), width=2)
            dr.text((x + 8, y - 8), ids[by[i][1]] if good else f'det {i}', fill=(255, 255, 0) if good else (255, 80, 80))
        dr.text((5, 5), f'Boresight RA {rb:.4f}  Dec {db:+.4f}  roll {roll:.2f}  ({len(matches)} matched)',
                fill=(255, 255, 255))
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        im.save(a.out)
        print(f'Annotated image saved to: {a.out}  (green=matched, red=unmatched detection, blue=predicted catalogue star)')

        if a.csv:
            Path(a.csv).parent.mkdir(parents=True, exist_ok=True)
            with open(a.csv, 'w', newline='') as f:
                wr = csv.writer(f)
                wr.writerow(['det', 'x', 'y', 'flux', 'id', 'ra_deg', 'dec_deg', 'mag', 'residual_arcsec'])
                for oi, ci, r in sorted(matches):
                    wr.writerow([oi, f"{ds[oi]['x']:.3f}", f"{ds[oi]['y']:.3f}", f"{ds[oi]['flux']:.1f}",
                                 ids[ci], f'{ra[ci]:.6f}', f'{de[ci]:.6f}', f'{mag[ci]:.2f}', f'{r * 3600:.2f}'])
            print(f'Matches written to: {a.csv}')
    except Exception as e:
        print(f'ERROR: {e}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
