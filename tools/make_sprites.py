#!/usr/bin/env python3
"""Generate the Star Merge pixel art.

Writes, relative to the project root:
  sprites/NN_name.png      64x64 sprite per body (used by the game)
  sprites/x4/NN_name.png   the same sprite upscaled 4x (nearest neighbour)
  sprites/sheet.png        preview sheet with the name under every body
  sprites/bg.png           low-res space background for the play field

Every body is rendered pixel by pixel: a sphere is shaded with a dithered,
quantised light ramp and coloured from small hand-picked palettes, so the
result stays crisp pixel art at any integer scale.
"""
import math
import os

import numpy as np
from PIL import Image

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
OUT = os.path.join(ROOT, 'sprites')

S = 64                 # sprite canvas size
C = (S - 1) / 2        # canvas centre
R = 22                 # body radius in pixels (the game maps this to the collision radius)

BAYER = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]) / 16.0
YY, XX = np.mgrid[0:S, 0:S].astype(float)
DITH = BAYER[YY.astype(int) % 4, XX.astype(int) % 4]
LIGHT = np.array([-0.55, -0.5, 0.67]) / np.linalg.norm([-0.55, -0.5, 0.67])


def hexc(h):
    return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], float) / 255


def pal(*hexes):
    return np.array([hexc(h) for h in hexes])


# ------------------------------------------------------------------ noise
def _hash(ix, iy, iz, seed):
    h = (ix * 73856093) ^ (iy * 19349663) ^ (iz * 83492791) ^ (seed * 2654435761)
    h = h.astype(np.uint64) & np.uint64(0xffffffff)
    h ^= h >> np.uint64(13)
    h = (h * np.uint64(1274126177)) & np.uint64(0xffffffff)
    h ^= h >> np.uint64(16)
    return (h & np.uint64(0xffff)).astype(float) / 65535.0


def vnoise(px, py, pz, seed):
    ix, iy, iz = (np.floor(a).astype(np.int64) for a in (px, py, pz))
    fx, fy, fz = px - ix, py - iy, pz - iz
    u, v, w = (f * f * (3 - 2 * f) for f in (fx, fy, fz))
    acc = 0.0
    for ox in (0, 1):
        for oy in (0, 1):
            for oz in (0, 1):
                wt = (u if ox else 1 - u) * (v if oy else 1 - v) * (w if oz else 1 - w)
                acc = acc + wt * _hash(ix + ox, iy + oy, iz + oz, seed)
    return acc


def fbm(p, freq, octaves=4, seed=0, stretch=(1, 1, 1)):
    total, amp, norm = 0.0, 1.0, 0.0
    for o in range(octaves):
        total = total + amp * vnoise(p[0] * freq * stretch[0] + o * 17.1,
                                     p[1] * freq * stretch[1] - o * 9.3,
                                     p[2] * freq * stretch[2] + o * 5.7, seed + o * 101)
        norm += amp
        amp *= 0.5
        freq *= 2
    return total / norm


# ------------------------------------------------------------------ geometry
def geom(rad=R, spin=0.0, tilt=0.0, lump=None):
    """Sphere of `rad` pixels: masks, view normal and surface coordinates."""
    dx, dy = (XX - C) / rad, (YY - C) / rad
    d2 = dx * dx + dy * dy
    if lump:
        amp, seed = lump
        a = np.arctan2(dy, dx)
        rr = 1 + amp * (vnoise(np.cos(a) * 2.2 + 5, np.sin(a) * 2.2 + 5, np.zeros_like(a), seed) - .5) * 2
        d2 = d2 / (rr * rr)
    inside = d2 <= 1.0
    z = np.sqrt(np.clip(1 - d2, 0, 1))
    ct, st = math.cos(tilt), math.sin(tilt)
    x1, y1 = dx * ct + dy * st, -dx * st + dy * ct
    cs, ss = math.cos(spin), math.sin(spin)
    x2, z2 = x1 * cs + z * ss, -x1 * ss + z * cs
    lat = np.arcsin(np.clip(-y1, -1, 1))
    lon = np.arctan2(x2, z2)
    return dict(rad=rad, dx=dx, dy=dy, d=np.sqrt(d2), inside=inside, z=z, p=(x2, y1, z2), lat=lat, lon=lon)


def pick(v, colors, edges):
    """Map a scalar field to discrete palette colours."""
    return colors[np.digitize(v, edges)]


def dith(v, amount=0.08):
    return v + (DITH - .5) * amount


def craters(g, n, seed, rmin, rmax):
    rs = np.random.default_rng(seed)
    floor = np.zeros((S, S), bool)
    rim = np.zeros((S, S), bool)
    px, py, pz = g['p']
    for _ in range(n):
        c = rs.normal(size=3)
        c /= np.linalg.norm(c)
        rad = rs.uniform(rmin, rmax)
        ang = np.arccos(np.clip(px * c[0] + py * c[1] + pz * c[2], -1, 1))
        floor |= ang < rad * .72
        rim |= (ang >= rad * .72) & (ang < rad)
    return floor & ~rim, rim & ~floor


def near(g, lon0, lat0, rad):
    """Mask of surface points within `rad` radians of (lon0, lat0)."""
    c = np.array([math.cos(lat0) * math.sin(lon0), -math.sin(lat0), math.cos(lat0) * math.cos(lon0)])
    px, py, pz = g['p']
    return np.arccos(np.clip(px * c[0] + py * c[1] + pz * c[2], -1, 1)) < rad


def ellipse(g, lon0, lat0, rx, ry):
    dl = (g['lon'] - lon0 + math.pi) % (2 * math.pi) - math.pi
    return (dl / rx) ** 2 + ((g['lat'] - lat0) / ry) ** 2 < 1


# ------------------------------------------------------------------ shading
def lit(g, albedo, levels=5, amb=.1, shadow=(.13, .11, .3)):
    lam = np.clip(g['dx'] * LIGHT[0] + g['dy'] * LIGHT[1] + g['z'] * LIGHT[2], 0, 1)
    s = amb + (1 - amb) * lam ** .85
    q = np.clip(np.floor(s * levels + DITH * .999) / levels, 0, 1)[..., None]
    dark = np.array(shadow)
    col = albedo * (dark + (1 - dark) * q)
    hi = (lam > .94)[..., None]
    return np.where(hi, np.minimum(1, col * 1.12 + .04), col)


def canvas():
    return np.zeros((S, S, 4))


def paint(img, mask, col, alpha=1.0):
    col = np.broadcast_to(col, (S, S, 3)) if np.ndim(col) == 1 else col
    a = np.broadcast_to(alpha, (S, S)) if np.ndim(alpha) == 0 else alpha
    m = mask & (a > 0)
    img[m, :3] = col[m]
    img[m, 3] = np.maximum(img[m, 3], a[m])
    return img


def body(img, g, albedo, **kw):
    return paint(img, g['inside'], lit(g, albedo, **kw))


def atmosphere(img, g, color, width=.09, alpha=.75):
    d = g['d']
    lam = np.clip(g['dx'] * LIGHT[0] + g['dy'] * LIGHT[1], -1, 1)
    m = (d > 1) & (d < 1 + width) & (lam > -.25) & ~(img[..., 3] > 0)
    paint(img, m, hexc(color), alpha)
    return img


def glow(img, g, color, width=.38, strength=.85):
    d = g['d']
    a = np.clip(1 - (d - 1) / width, 0, 1) ** 1.4
    a = np.floor(a * 4 + DITH * .999) / 4 * strength       # stepped, dithered falloff
    m = (d > 1) & (a > 0) & (img[..., 3] == 0)
    paint(img, m, hexc(color), a)
    return img


def spikes(img, color, length=31, width=0.6, strength=.9):
    dx, dy = XX - C, YY - C
    for m, dist in ((np.abs(dy) < width, np.abs(dx)), (np.abs(dx) < width, np.abs(dy))):
        a = np.clip(1 - dist / length, 0, 1) * strength
        a = np.floor(a * 4 + DITH * .999) / 4
        mm = m & (a > 0) & (img[..., 3] < a)
        img[mm, :3] = hexc(color)
        img[mm, 3] = a[mm]
    return img


def ring(img, g, r1, r2, incl, angle, colors_fn):
    """Planetary ring in the planet's radius units; the far half hides behind the planet."""
    dx, dy = g['dx'], g['dy']
    ca, sa = math.cos(angle), math.sin(angle)
    u, v = dx * ca + dy * sa, -dx * sa + dy * ca
    rr = np.sqrt(u * u + (v / math.sin(incl)) ** 2)
    m = (rr > r1) & (rr < r2) & (~g['inside'] | (v > 0))
    col, a = colors_fn(rr, u)
    paint(img, m & (a > 0), col, a)
    return img


def star(img, g, colors, glow_col, freq=5.0, seed=1, cells=.45, spots=0, glow_w=.38, glow_s=.85):
    limb = g['z'] ** .5
    gran = fbm(g['p'], freq, 4, seed)
    t = dith((1 - cells) * limb + cells * gran, .12)
    edges = np.linspace(.3, .85, len(colors) - 1)
    col = pick(t, colors, edges)
    if spots:
        fl, _ = craters(g, spots, seed + 7, .05, .12)
        col = np.where((fl & (g['z'] > .35))[..., None], colors[0] * .6, col)
    paint(img, g['inside'], col)
    glow(img, g, glow_col, glow_w, glow_s)
    return img


def prominence(img, g, ang, color, size=.22):
    ex, ey = math.cos(ang) * g['rad'], math.sin(ang) * g['rad']
    dd = np.hypot(XX - C - ex, YY - C - ey) / g['rad']
    m = (np.abs(dd - size) < .05) & (g['d'] > 1)
    paint(img, m, hexc(color))


# ------------------------------------------------------------------ bodies
def ceres():
    g = geom(lump=(.07, 3))
    p = g['p']
    col = pick(dith(fbm(p, 2.2, 4, 11)), pal('#5f574e', '#7c7368', '#968c7f', '#aaa092'), [.42, .5, .58])
    fl, rim = craters(g, 22, 4, .1, .3)
    col = np.where(fl[..., None], col * .78, col)
    col = np.where(rim[..., None], np.minimum(1, col * 1.18), col)
    col = np.where(near(g, .25, .2, .07)[..., None], hexc('#f6f6ee'), col)
    return body(canvas(), g, col)


def pluto():
    g = geom(spin=.1)
    p, lon, lat = g['p'], g['lon'], g['lat']
    col = pick(dith(fbm(p, 2.0, 4, 21)), pal('#a87a55', '#c49a72', '#d9b892', '#e8cfaf'), [.4, .5, .6])
    dark = (lat > -.45) & (lat < .05) & (lon < -.25) & (lon > -2.2) & (fbm(p, 3, 3, 22) > .38)
    col = np.where(dark[..., None], hexc('#5c2f22'), col)
    hx, hy = (lon - .5) / .5, (lat + .1) / .48
    heart = (hx * hx + hy * hy - 1) ** 3 - hx * hx * hy ** 3 <= 0
    col = np.where(heart[..., None], hexc('#f5eee4'), col)
    return body(canvas(), g, col)


def moon():
    g = geom(spin=-.2)
    p = g['p']
    col = pick(dith(fbm(p, 2.6, 4, 31)), pal('#9c9c97', '#b4b4ae', '#c9c9c3', '#dadad4'), [.4, .5, .6])
    maria = fbm(p, 1.4, 3, 32) > .54
    col = np.where(maria[..., None], pick(dith(fbm(p, 4, 2, 33)), pal('#6d6d6a', '#7d7d79'), [.5]), col)
    fl, rim = craters(g, 16, 34, .07, .22)
    col = np.where(fl[..., None], col * .8, col)
    col = np.where(rim[..., None], np.minimum(1, col * 1.15), col)
    return body(canvas(), g, col)


def mercury():
    g = geom(spin=.6)
    p = g['p']
    col = pick(dith(fbm(p, 2.4, 4, 41)), pal('#6f6157', '#87786b', '#9f9081', '#b5a696'), [.4, .5, .6])
    fl, rim = craters(g, 38, 42, .06, .2)
    col = np.where(fl[..., None], col * .76, col)
    col = np.where(rim[..., None], np.minimum(1, col * 1.2), col)
    return body(canvas(), g, col)


def mars():
    g = geom(spin=.3, tilt=.15)
    p, lat, lon = g['p'], g['lat'], g['lon']
    col = pick(dith(fbm(p, 2.0, 5, 51)), pal('#8a3520', '#b04a2a', '#c95d34', '#dd7a4a'), [.42, .5, .58])
    valles = (np.abs(lat + .12 + .05 * np.sin(lon * 3)) < .045) & (lon > -.9) & (lon < .5)
    col = np.where(valles[..., None], hexc('#6a2616'), col)
    cap = lat + .06 * (fbm(p, 4, 2, 52) - .5) > .8
    cap |= lat < -.88
    col = np.where(cap[..., None], hexc('#f3eee8'), col)
    img = body(canvas(), g, col)
    return atmosphere(img, g, '#e8a07a', .06, .45)


def venus():
    g = geom(spin=.4)
    p = g['p']
    v = fbm((p[0] + .3 * np.sin(p[1] * 6), p[1], p[2]), 1.6, 5, 61, stretch=(1, 4, 1))
    col = pick(dith(v), pal('#b88a45', '#cfa45a', '#e2bf78', '#efd49a', '#f8e7bd'), [.38, .46, .54, .62])
    img = body(canvas(), g, col)
    return atmosphere(img, g, '#ffe7a8', .08, .5)


def earth():
    g = geom(spin=-.4, tilt=.2)
    p, lat = g['p'], g['lat']
    h = fbm(p, 1.5, 5, 71)
    land = h > .52
    col = np.where((h > .48)[..., None], hexc('#3f8ee6'), hexc('#245bc4'))
    lc = pick(dith(h), pal('#4f9e3f', '#3a8634', '#2b6d2c'), [.58, .66])
    desert = (np.abs(lat) < .5) & (fbm(p, 2.5, 3, 72) > .56)
    lc = np.where(desert[..., None], hexc('#d6b56b'), lc)
    col = np.where(land[..., None], lc, col)
    ice = np.abs(lat) + .08 * (fbm(p, 4, 2, 73) - .5) > 1.2
    col = np.where(ice[..., None], hexc('#eef6ff'), col)
    cl = dith(fbm(p, 2.0, 4, 74, stretch=(1, 2, 1)), .1)
    col = np.where((cl > .59)[..., None], hexc('#ffffff'), col)
    img = body(canvas(), g, col)
    return atmosphere(img, g, '#8fc4ff', .09, .7)


def kepler22b():
    g = geom(spin=.7, tilt=-.25)
    p = g['p']
    col = pick(dith(fbm(p, 1.8, 4, 81, stretch=(1, 2.5, 1))), pal('#0b4a66', '#11698a', '#1a8aa6', '#2aa9bb'), [.42, .5, .58])
    px, py, pz = p
    a = np.arctan2(py + .25, px - .15)
    rr = np.hypot(px - .15, py + .25)
    swirl = (np.sin(a * 2 + rr * 18) > .55) & (rr < .45) & (rr > .06)
    cl = (fbm(p, 2.4, 4, 82, stretch=(1, 3, 1)) > .6) | swirl
    col = np.where(cl[..., None], hexc('#e6fbff'), col)
    img = body(canvas(), g, col)
    return atmosphere(img, g, '#7fe6ff', .09, .7)


def neptune():
    g = geom(tilt=.15)
    p, lat = g['p'], g['lat']
    v = .5 + .5 * np.sin(lat * 7 + 2.2 * fbm(p, 2, 3, 91, stretch=(1, 3, 1)))
    col = pick(dith(v), pal('#1f3aa8', '#2b52cf', '#3e6ce8', '#5b8afa'), [.3, .6, .85])
    col = np.where(ellipse(g, .3, -.3, .32, .16)[..., None], hexc('#15246e'), col)
    streak = ellipse(g, .45, -.47, .3, .05) | ellipse(g, -.4, .38, .45, .045)
    col = np.where(streak[..., None], hexc('#e3ecff'), col)
    img = body(canvas(), g, col)
    return atmosphere(img, g, '#6f9cff', .07, .55)


def uranus():
    rad = 19
    g = geom(rad, tilt=1.35)
    p, lat = g['p'], g['lat']
    v = .5 + .5 * np.sin(lat * 4 + .8 * fbm(p, 2, 3, 101))
    col = pick(dith(v, .06), pal('#83cfd6', '#97dcdf', '#ace8e8'), [.4, .75])
    col = np.where((lat > 1.1)[..., None], hexc('#c4f2f0'), col)
    img = body(canvas(), g, col)
    atmosphere(img, g, '#b8f4ff', .07, .5)

    def rc(rr, u):
        a = np.where((rr > 1.5) & (rr < 1.56) | (rr > 1.48) & (rr < 1.5), .85, 0.0)
        a = np.where((rr > 1.36) & (rr < 1.4), .5, a)
        return np.broadcast_to(hexc('#d8f2f6'), (S, S, 3)), a
    return ring(img, g, 1.3, 1.6, .34, 1.35, rc)


def saturn():
    rad = 15
    g = geom(rad, tilt=-.3)
    p, lat = g['p'], g['lat']
    v = .5 + .5 * np.sin(lat * 9 + .9 * fbm(p, 2, 3, 111, stretch=(1, 3, 1)))
    col = pick(dith(v), pal('#b8925a', '#cfaa6c', '#e2c58a', '#efdcae'), [.3, .55, .8])
    col = np.where((np.abs(lat) > 1.15)[..., None], hexc('#9c8a6a'), col)
    img = body(canvas(), g, col)

    def rc(rr, u):
        c = np.zeros((S, S, 3))
        a = np.zeros((S, S))
        for lo, hi, hx, al in ((1.22, 1.5, '#9a8466', .75), (1.5, 1.8, '#ead9b0', 1), (1.88, 2.06, '#cbb68c', 1)):
            m = (rr >= lo) & (rr < hi)
            c[m] = hexc(hx)
            a[m] = al
        band = (np.floor(rr * 30) % 3 == 0) & (DITH > .5)
        c = np.where(band[..., None], c * .9, c)
        return c, a
    return ring(img, g, 1.22, 2.06, .4, -.3, rc)


def jupiter():
    g = geom(tilt=.08)
    p, lat = g['p'], g['lat']
    t = lat * 6.5 + 1.3 * (fbm(p, 2, 4, 121, stretch=(.8, 4, .8)) - .5)
    bands = pal('#efe2c6', '#c48a58', '#f3e8d0', '#ad6c42', '#e6cb9e', '#9e5e38')
    col = bands[np.floor(t + 50).astype(int) % len(bands)]
    grs = ellipse(g, .35, -.36, .42, .19)
    core = ellipse(g, .35, -.36, .3, .12)
    rim = grs & ~ellipse(g, .35, -.36, .36, .16)
    col = np.where(grs[..., None], hexc('#c4523a'), col)
    col = np.where(core[..., None], hexc('#de7353'), col)
    col = np.where(rim[..., None], hexc('#f5dcc0'), col)
    return body(canvas(), g, col)


def hd189733b():
    g = geom(tilt=-.2, spin=.5)
    p, lat = g['p'], g['lat']
    t = lat * 5 + 1.6 * fbm(p, 2, 4, 131, stretch=(.8, 3.5, .8))
    bands = pal('#1a3ca6', '#2c62d4', '#142f88', '#3e7ae8', '#2147b4')
    col = bands[np.floor(t + 50).astype(int) % len(bands)]
    storm = fbm(p, 3, 3, 132) > .64
    col = np.where(storm[..., None], hexc('#9ec2ff'), col)
    img = body(canvas(), g, col)
    return atmosphere(img, g, '#6aa8ff', .12, .6)


def proxima():
    g = geom()
    img = star(canvas(), g, pal('#6e1a10', '#a12c18', '#cf4a24', '#f06c34', '#ff9a5a'), '#ff4a20',
               freq=6, seed=141, spots=5, glow_w=.32)
    prominence(img, g, -2.4, '#ff6a30', .2)
    return img


def sun():
    g = geom()
    img = star(canvas(), g, pal('#cf5f17', '#ec861d', '#ffae2b', '#ffd456', '#fff0a6'), '#ffb028',
               freq=7, seed=151, spots=4, glow_w=.42)
    prominence(img, g, -2.2, '#ff7a28', .22)
    prominence(img, g, .5, '#ff8a30', .16)
    return img


def sirius():
    g = geom()
    img = star(canvas(), g, pal('#8aa6f5', '#b4c8ff', '#dbe6ff', '#f4f8ff', '#ffffff'), '#a8c4ff',
               freq=5, seed=161, cells=.25, glow_w=.42)
    spikes(img, '#dbe6ff', 31, .6, .95)
    b = np.hypot(XX - (C + 23), YY - (C - 22)) < 2.2          # little companion: Sirius B
    paint(img, b, hexc('#ffffff'))
    return img


def arcturus():
    g = geom()
    return star(canvas(), g, pal('#a14c12', '#cc6f22', '#ea9238', '#fbb860', '#ffd894'), '#ff9530',
                freq=4.5, seed=171, cells=.5, glow_w=.4)


def rigel():
    g = geom()
    img = star(canvas(), g, pal('#4a6ff0', '#7898ff', '#a8c0ff', '#d6e2ff', '#f6f9ff'), '#5f8cff',
               freq=5, seed=181, cells=.3, glow_w=.45, glow_s=.95)
    spikes(img, '#b8ccff', 31, .7, 1)
    return img


def betelgeuse():
    g = geom(lump=(.06, 191))
    img = star(canvas(), g, pal('#5c120c', '#8c2116', '#ba331d', '#e0552a', '#ff8446'), '#ff3e20',
               freq=2.2, seed=192, cells=.62, glow_w=.33, glow_s=.7)
    return img


def black_hole():
    img = canvas()
    dx, dy = (XX - C) / R, (YY - C) / R
    ang = -.12
    ca, sa = math.cos(ang), math.sin(ang)
    u, v = dx * ca + dy * sa, -dx * sa + dy * ca
    sd = np.hypot(dx, dy)
    incl = .22
    rr = np.hypot(u, v / math.sin(incl))
    disk_cols = pal('#fff6d6', '#ffd47c', '#ff9c3c', '#e2562a', '#9a2c1a')

    def disk_color(r, doppler):
        t = dith((r - .62) / .8 - doppler * .3, .12)
        return pick(t, disk_cols, [.12, .32, .55, .8])

    disk = (rr > .62) & (rr < 1.42)
    back = disk & (v <= 0)
    paint(img, back, disk_color(rr, -u / np.maximum(rr, .01)))
    # light from the far side of the disk, bent over the top and under the bottom of the hole
    lens = (sd > .58) & (sd < .9) & (v < 0)
    lens_lo = (sd > .58) & (sd < .7) & (v > 0)
    paint(img, lens | lens_lo, disk_color(.62 + (sd - .58) * 1.6, -u / np.maximum(sd, .01)))
    paint(img, sd <= .52, np.zeros(3))
    paint(img, (sd > .52) & (sd < .6), hexc('#fff3c4'))
    front = disk & (v > 0)
    paint(img, front, disk_color(rr, -u / np.maximum(rr, .01)))
    return img


BODIES = [
    ('Ceres', ceres), ('Pluto', pluto), ('Moon', moon), ('Mercury', mercury), ('Mars', mars),
    ('Venus', venus), ('Earth', earth), ('Kepler-22b', kepler22b), ('Neptune', neptune),
    ('Uranus', uranus), ('Saturn', saturn), ('Jupiter', jupiter), ('HD 189733b', hd189733b),
    ('Proxima Centauri', proxima), ('Sun', sun), ('Sirius', sirius), ('Arcturus', arcturus),
    ('Rigel', rigel), ('Betelgeuse', betelgeuse), ('Black Hole', black_hole),
]


# ------------------------------------------------------------------ pixel font (5x7)
FONT = {
    'A': ['.###.', '#...#', '#...#', '#####', '#...#', '#...#', '#...#'],
    'B': ['####.', '#...#', '#...#', '####.', '#...#', '#...#', '####.'],
    'C': ['.###.', '#...#', '#....', '#....', '#....', '#...#', '.###.'],
    'D': ['####.', '#...#', '#...#', '#...#', '#...#', '#...#', '####.'],
    'E': ['#####', '#....', '#....', '####.', '#....', '#....', '#####'],
    'F': ['#####', '#....', '#....', '####.', '#....', '#....', '#....'],
    'G': ['.###.', '#...#', '#....', '#.###', '#...#', '#...#', '.####'],
    'H': ['#...#', '#...#', '#...#', '#####', '#...#', '#...#', '#...#'],
    'I': ['.###.', '..#..', '..#..', '..#..', '..#..', '..#..', '.###.'],
    'J': ['..###', '...#.', '...#.', '...#.', '...#.', '#..#.', '.##..'],
    'K': ['#...#', '#..#.', '#.#..', '##...', '#.#..', '#..#.', '#...#'],
    'L': ['#....', '#....', '#....', '#....', '#....', '#....', '#####'],
    'M': ['#...#', '##.##', '#.#.#', '#.#.#', '#...#', '#...#', '#...#'],
    'N': ['#...#', '#...#', '##..#', '#.#.#', '#..##', '#...#', '#...#'],
    'O': ['.###.', '#...#', '#...#', '#...#', '#...#', '#...#', '.###.'],
    'P': ['####.', '#...#', '#...#', '####.', '#....', '#....', '#....'],
    'Q': ['.###.', '#...#', '#...#', '#...#', '#.#.#', '#..#.', '.##.#'],
    'R': ['####.', '#...#', '#...#', '####.', '#.#..', '#..#.', '#...#'],
    'S': ['.####', '#....', '#....', '.###.', '....#', '....#', '####.'],
    'T': ['#####', '..#..', '..#..', '..#..', '..#..', '..#..', '..#..'],
    'U': ['#...#', '#...#', '#...#', '#...#', '#...#', '#...#', '.###.'],
    'V': ['#...#', '#...#', '#...#', '#...#', '#...#', '.#.#.', '..#..'],
    'W': ['#...#', '#...#', '#...#', '#.#.#', '#.#.#', '#.#.#', '.#.#.'],
    'X': ['#...#', '#...#', '.#.#.', '..#..', '.#.#.', '#...#', '#...#'],
    'Y': ['#...#', '#...#', '.#.#.', '..#..', '..#..', '..#..', '..#..'],
    'Z': ['#####', '....#', '...#.', '..#..', '.#...', '#....', '#####'],
    '0': ['.###.', '#...#', '#..##', '#.#.#', '##..#', '#...#', '.###.'],
    '1': ['..#..', '.##..', '..#..', '..#..', '..#..', '..#..', '.###.'],
    '2': ['.###.', '#...#', '....#', '...#.', '..#..', '.#...', '#####'],
    '3': ['#####', '...#.', '..#..', '...#.', '....#', '#...#', '.###.'],
    '4': ['...#.', '..##.', '.#.#.', '#..#.', '#####', '...#.', '...#.'],
    '5': ['#####', '#....', '####.', '....#', '....#', '#...#', '.###.'],
    '6': ['..##.', '.#...', '#....', '####.', '#...#', '#...#', '.###.'],
    '7': ['#####', '....#', '...#.', '..#..', '.#...', '.#...', '.#...'],
    '8': ['.###.', '#...#', '#...#', '.###.', '#...#', '#...#', '.###.'],
    '9': ['.###.', '#...#', '#...#', '.####', '....#', '...#.', '.##..'],
    '-': ['.....', '.....', '.....', '#####', '.....', '.....', '.....'],
    ' ': ['.....'] * 7,
}


def text_width(s, sc):
    return (len(s) * 6 - 1) * sc


def draw_text(arr, s, x, y, sc, color, shadow=(20, 14, 40)):
    for off, col in (((sc, sc), shadow), ((0, 0), color)):
        cx = x + off[0]
        for ch in s.upper():
            for r, row in enumerate(FONT.get(ch, FONT[' '])):
                for c, bit in enumerate(row):
                    if bit == '#':
                        y0, x0 = y + off[1] + r * sc, cx + c * sc
                        arr[y0:y0 + sc, x0:x0 + sc, :3] = col
                        arr[y0:y0 + sc, x0:x0 + sc, 3] = 255
            cx += 6 * sc


# ------------------------------------------------------------------ background
def starfield(w, h, seed, nebula=True, calm=False):
    """Space backdrop; `calm` keeps it quiet so it never competes with the bodies."""
    rs = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    dth = BAYER[yy.astype(int) % 4, xx.astype(int) % 4]
    t = yy / h
    sky = pal('#07051a', '#0a0822', '#0d0a2a', '#100c32') if calm else pal('#07051a', '#0d0a2a', '#150f3c', '#1d1450')
    img = sky[np.clip(np.floor(t * 3.2 + dth * .999), 0, 3).astype(int)]
    if nebula:
        n = fbm((xx / 60, yy / 60, np.zeros_like(xx) + .5), 1.0, 5, seed + 1)
        n2 = fbm((xx / 45, yy / 45, np.zeros_like(xx) + 3.5), 1.0, 4, seed + 2)
        lvl = np.floor((n - .54) * 8 + dth * .999)
        neb = pal('#1c1248', '#281858', '#352066', '#442873')
        if calm:
            lvl = np.floor((n - .58) * 5 + dth * .999)
            neb = pal('#110c30', '#150f38', '#191240', '#1c1446')
        m = lvl >= 0
        img[m] = neb[np.clip(lvl[m].astype(int), 0, 3)]
        lvl2 = np.floor((n2 - .6) * 8 + dth * .999)
        neb2 = pal('#11294a', '#173a5a', '#1e4b68')
        if calm:
            lvl2 = np.floor((n2 - .64) * 5 + dth * .999)
            neb2 = pal('#0c1634', '#0f1c3c', '#122242')
        m2 = lvl2 >= 0
        img[m2] = neb2[np.clip(lvl2[m2].astype(int), 0, 2)]
    out = (img * 255).astype(np.uint8)
    cols = [(255, 255, 255), (190, 210, 255), (255, 236, 180), (255, 190, 190)]
    if calm:
        cols = [(150, 150, 190), (120, 135, 180), (165, 155, 135), (95, 95, 140)]
    for _ in range(int(w * h / (160 if calm else 55))):
        x, y = rs.integers(0, w), rs.integers(0, h)
        out[y, x] = cols[rs.integers(0, 4)] if rs.random() < .6 else (140, 140, 190) if not calm else (70, 70, 110)
    for _ in range(0 if calm else int(w * h / 1600)):
        x, y = rs.integers(2, w - 2), rs.integers(2, h - 2)
        c = cols[rs.integers(0, 3)]
        out[y, x] = c
        for ddx, ddy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            out[y + ddy, x + ddx] = (np.array(c) * .55).astype(np.uint8)
    return out


def to_img(a):
    return Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8), 'RGBA')


def slug(name):
    return name.lower().replace(' ', '_').replace('-', '_')


def main():
    os.makedirs(os.path.join(OUT, 'x4'), exist_ok=True)
    sprites = []
    for i, (name, fn) in enumerate(BODIES):
        im = to_img(fn())
        fname = f'{i + 1:02d}_{slug(name)}.png'
        im.save(os.path.join(OUT, fname))
        im.resize((S * 4, S * 4), Image.NEAREST).save(os.path.join(OUT, 'x4', fname))
        sprites.append((name, im))
        print('wrote', fname)

    # preview sheet: 5 x 4 grid, sprite at 3x with its name underneath
    cols, rows, sc, name_sc = 5, 4, 3, 2
    cw, chh, top = 300, S * sc + 52, 96
    sheet_w, sheet_h = cols * cw, top + rows * chh + 20
    bg = starfield(sheet_w // 2, sheet_h // 2, 7, nebula=True)
    sheet = np.array(Image.fromarray(bg).resize((sheet_w, sheet_h), Image.NEAREST).convert('RGBA'))
    title = 'STAR MERGE - 20 BODIES'
    draw_text(sheet, title, (sheet_w - text_width(title, 4)) // 2, 30, 4, (255, 214, 102))
    for i, (name, im) in enumerate(sprites):
        cx, cy = (i % cols) * cw, top + (i // cols) * chh
        big = np.array(im.resize((S * sc, S * sc), Image.NEAREST))
        x0, y0 = cx + (cw - S * sc) // 2, cy
        a = big[..., 3:4] / 255.0
        region = sheet[y0:y0 + S * sc, x0:x0 + S * sc]
        region[..., :3] = (big[..., :3] * a + region[..., :3] * (1 - a)).astype(np.uint8)
        label = name.upper()
        num = f'{i + 1:02d}'
        draw_text(sheet, num, cx + 12, cy + 8, 2, (150, 160, 220))
        draw_text(sheet, label, cx + (cw - text_width(label, name_sc)) // 2, cy + S * sc + 12, name_sc, (255, 255, 255))
    Image.fromarray(sheet).save(os.path.join(OUT, 'sheet.png'))
    print('wrote sheet.png')

    Image.fromarray(starfield(170, 262, 3, calm=True)).save(os.path.join(OUT, 'bg.png'))
    print('wrote bg.png')


if __name__ == '__main__':
    main()
