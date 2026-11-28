# -*- coding: utf-8 -*-
"""Procedural live wallpaper for assets/gothic-lolita.jpg (no AI repaint, pure geometry).

Layers, back -> front, each with its own motion and material "feel":
  0  background plate  - never moves (pixels stay identical to the source)
  1  body              - breathing (chest/shoulders rise), cuff & sleeve-frill sway
  2  lace skirt        - whole ruffles sway in a slow gust, free edges move most
  3  accessories       - jewelled chain/pendant (rigid metal, slow lagging swing),
                         black ribbons (light fabric, fast swing that bends toward the tips)
  4  head (+hand)      - rigid tiny tilt, rides on the breath; eyes blink and glance aside;
                         hair: braids/bangs/strands with several overlapping frequencies so
                         neighbouring strands move independently
                         face: a slow smile (mouth corners lift & widen, cheeks push the lower
                         lids up a touch), a lopsided smirk while she glances aside, a brief
                         lip press, and corners that breathe with her
Uncovered slivers behind a moving layer are filled from an inpainted plate of the layer below.

deps : opencv-python numpy (+ rembg[cpu] only if the cached masks are missing), ffmpeg
usage: python scripts/animate.py assets/gothic-lolita.jpg out_dir [scale=1.6] [workers=2]
       -> out_dir/gothic-lolita.mp4 (2560x1440@60, 8 s loop) + gothic-lolita.webp (1280@24)
"""
import sys, os, subprocess, numpy as np, cv2
from multiprocessing import Pool

SRC, OUT = sys.argv[1], sys.argv[2]
K = float(sys.argv[3]) if len(sys.argv) > 3 else 1.6
WORKERS = int(sys.argv[4]) if len(sys.argv) > 4 else 2
os.makedirs(OUT, exist_ok=True)
FPS, DUR = 60, 8.0                       # 8 s loop: every period below divides 8 s exactly
N = int(FPS * DUR)
TAU = 2 * np.pi

img = cv2.imread(SRC).astype(np.float32)
H, W = img.shape[:2]
gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
GRAY = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)

def poly(pts):
    m = np.zeros((H, W), np.uint8); cv2.fillPoly(m, [np.array(pts, np.int32)], 1); return m

def soft_poly(pts, sigma):
    return cv2.GaussianBlur(poly(pts).astype(np.float32), (0, 0), sigma)

def smooth(x):
    x = np.clip(x, 0, 1); return x * x * (3 - 2 * x)

def aa(mask):                             # hard mask with a ~1 px anti-aliased edge only
    m = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 0.7); m[m < 0.04] = 0; return m

def ring(mask, out_px, in_px):            # band around a mask edge (for inpainting holes)
    m = (mask > 0.02).astype(np.uint8)
    o = cv2.dilate(m, np.ones((2 * out_px + 1,) * 2, np.uint8))
    i = cv2.erode(m, np.ones((2 * in_px + 1,) * 2, np.uint8))
    return (o & (1 - i)).astype(np.uint8)

def rot_field(ang, piv):                  # dest->src offsets of a rotation by ang about piv
    rx, ry = gx - piv[0], gy - piv[1]
    c, s = np.cos(ang), np.sin(ang)
    return c * rx - s * ry - rx, s * rx + c * ry - ry

# ======================= eyes: blink + gaze =======================
# y_u = lower edge of the upper lash band, y_l = lower lid line
EYES = [
    dict(up=[(564, 289), (575, 285), (590, 285), (603, 287), (613, 292), (620, 300), (628, 311)],
         lo=[(564, 289), (572, 300), (583, 313), (598, 317), (612, 316), (620, 314), (628, 311)],
         lash=5, skin=12),
    dict(up=[(697, 329), (705, 322), (715, 318), (735, 316), (754, 317), (766, 319), (773, 322)],
         lo=[(697, 329), (710, 335), (726, 341), (741, 342), (760, 339), (773, 322)],
         lash=5, skin=12),
]

def blink_map(t):
    """dest->src vertical map at closure t: upper lid slides down over the eye (occlusion)."""
    my = gy.copy()
    if t <= 0: return my
    for e in EYES:
        ux, uy = np.array(e['up'], np.float32).T
        lx, ly = np.array(e['lo'], np.float32).T
        L, S = e['lash'], e['skin']
        for x in range(int(ux.min()), int(ux.max()) + 1):
            yu = np.interp(x, ux, uy); yl = np.interp(x, lx, ly)
            yn = yu + t * max(yl - yu - 0.5, 0)
            ytop = yu - L - S
            ys = np.arange(int(ytop), int(np.ceil(yn)) + 1, dtype=np.float32)
            src = ys.copy()
            lash = ys >= yn - L
            src[lash] = ys[lash] - (yn - yu)
            sk = (~lash) & (ys >= ytop)
            src[sk] = ytop + (ys[sk] - ytop) * (yu - L - ytop) / max(yn - L - ytop, 1e-3)
            my[ys.astype(int), x] = src
    return my

# gaze: horizontal shift of everything strictly between the lids (iris + sclera), tapered
# toward the eye corners so the white absorbs the shift; lids and lashes never move.
GAZE_W = np.zeros((H, W), np.float32)
for e in EYES:
    ux, uy = np.array(e['up'], np.float32).T
    lx, ly = np.array(e['lo'], np.float32).T
    x0, x1 = int(ux.min()), int(ux.max())
    for x in range(x0, x1 + 1):
        yu = np.interp(x, ux, uy); yl = np.interp(x, lx, ly)
        taper = smooth((x - x0) / 12.0) * smooth((x1 - x) / 12.0)
        GAZE_W[int(np.ceil(yu + 0.5)):int(yl - 0.5) + 1, x] = taper

# ======================= mouth / cheeks: expression =======================
# The lips are closed line art, so every expression is a small, smooth warp of what is already
# drawn (no repaint). Corner points traced on the source: left (616,390), right (689,404),
# lower lip centre (650,409). Offsets are dest->src: dy = +u moves the feature UP by u px.
# Summed peak gradient stays < 0.75 px/px, so the warp never folds.
def gbump(cx, cy, sx, sy):
    return np.exp(-(((gx - cx) / sx) ** 2 + ((gy - cy) / sy) ** 2) / 2).astype(np.float32)

MCL, MCR = gbump(616, 390, 10, 7), gbump(689, 404, 10, 7)          # mouth corners
LIP_LO = gbump(651, 410, 16, 5.5)                                     # lower lip body
CHEEK_L, CHEEK_R = gbump(596, 330, 16, 9), gbump(738, 352, 20, 10)    # just under the lower lids
ZERO = np.zeros((H, W), np.float32)
# (dx, dy) per unit of each expression channel; sized to read at README width (~830 px,
# i.e. about half of source scale), where the whole mouth is only ~40 px wide
SMILE = (2.4 * MCL - 2.6 * MCR,                                        # corners widen outward
         4.8 * MCL + 5.2 * MCR + 0.6 * LIP_LO + 1.4 * CHEEK_L + 1.6 * CHEEK_R)
SMIRK = (-1.0 * MCR, 2.6 * MCR)                                        # right corner only
PRESS = (-1.3 * MCL + 1.3 * MCR,                                       # corners draw in
         2.0 * LIP_LO + 0.6 * MCL + 0.6 * MCR)                         # lower lip rolls up
CORNER = (ZERO, 0.7 * MCL + 0.7 * MCR)                                 # tiny breath-linked lift

def env(t, a, b, rise, fall):
    """0 -> 1 -> 0 envelope: eases in over [a, a+rise], out over [b-fall, b]."""
    if t <= a or t >= b: return 0.0
    return smooth((t - a) / rise) * (1 - smooth((t - (b - fall)) / fall))

def face_field(t, b):
    """Expression timeline, all channels 0 at t = 0 and t = 8 s so the loop stays seamless.
    1.2 s  a smile grows; it deepens into a lopsided smirk while she glances aside (2.4-4.2 s);
    5.0 s  it fades; 5.35 s a brief lip press ("hm"); corners breathe with her throughout."""
    smile = env(t, 1.2, 5.3, 0.7, 0.8)
    smirk = env(t, 2.45, 4.5, 0.35, 0.5)
    press = env(t, 5.35, 6.0, 0.18, 0.3)
    w = ((SMILE, smile), (SMIRK, smirk), (PRESS, press), (CORNER, b))
    return sum(f[0] * k for f, k in w), sum(f[1] * k for f, k in w)

# ======================= head (+ hand): rigid, hard cut at the jaw =======================
HEAD_POLY = [(395, 60), (470, 42), (560, 40), (700, 40), (830, 44), (950, 58), (1022, 140),
             (1035, 260), (1015, 360), (960, 388), (905, 372), (880, 345), (840, 330),
             (800, 350), (790, 375), (757, 418), (720, 431), (680, 440), (660, 447), (642, 446),
             (640, 470), (636, 490), (600, 472), (560, 452), (530, 440), (522, 352),
             (420, 325), (388, 200)]
HAND_POLY = [(488, 212), (562, 212), (566, 300), (592, 330), (622, 380), (646, 445), (642, 496),
             (600, 474), (560, 454), (528, 442), (498, 422), (482, 340)]
_hh = poly(HEAD_POLY) | poly(HAND_POLY)
HEADHARD = _hh.astype(np.float32)
HEAD = smooth(1 - cv2.distanceTransform(1 - _hh, cv2.DIST_L2, 5) / 45.0)  # 1 inside
ARM_ZONE = poly([(470, 430), (530, 440), (560, 452), (600, 472), (636, 490), (645, 600), (470, 600)])
STRAND_ZONE = poly([(790, 375), (800, 350), (840, 330), (875, 345), (880, 440), (780, 440)])
PIV = (600.0, 465.0)                      # chin resting on the hand

BRAID_L = soft_poly([(525, 55), (565, 105), (550, 200), (535, 300), (485, 330), (415, 300),
                     (398, 200), (415, 95), (465, 50)], 14)
BRAID_R = soft_poly([(835, 55), (945, 65), (1005, 150), (1020, 255), (1005, 340), (960, 370),
                     (895, 335), (865, 245), (845, 150)], 14)
STRAND_R = soft_poly([(805, 270), (850, 270), (865, 440), (812, 440)], 6)
BANGS = soft_poly([(575, 110), (790, 110), (800, 270), (575, 260)], 12)
_hd = np.clip(cv2.distanceTransform(1 - poly(HAND_POLY), cv2.DIST_L2, 5) / 25.0, 0, 1)
HAND_FREE = smooth(_hd).astype(np.float32)                 # no hair motion on the fingers

def hair_field(t):
    """Hair = slow, heavy, several overlapping waves (periods 8/3 s, 2 s, 1.6 s) whose phase
    varies along AND across the hair, so neighbouring strands do not move in lock-step."""
    w1, w2, w3 = TAU * t / (8 / 3), TAU * t / 2.0, TAU * t / 1.6
    dx = np.zeros((H, W), np.float32); dy = np.zeros((H, W), np.float32)
    d = np.hypot(gx - 540, gy - 85); a = np.clip((d - 25) / 230, 0, 1)
    dx += BRAID_L * a * (7.5 * np.sin(w1 - d / 140) + 2.5 * np.sin(w2 - d / 55 + 0.7))
    dy += BRAID_L * a * (2.2 * np.sin(w1 - d / 140 + 1.2) + 1.0 * np.sin(w3 - d / 60))
    d = np.hypot(gx - 860, gy - 85); a = np.clip((d - 25) / 260, 0, 1)
    dx += BRAID_R * a * (8.0 * np.sin(w1 - d / 150 + 0.9) + 2.5 * np.sin(w2 - d / 55 + 2.1))
    dy += BRAID_R * a * (2.2 * np.sin(w1 - d / 150 + 2.0) + 1.0 * np.sin(w3 - d / 60 + 1.0))
    a = np.clip((gy - 270) / 160, 0, 1)
    dx += STRAND_R * a * (4.5 * np.sin(w1 - gy / 90 + 0.3) + 1.5 * np.sin(w2 - gy / 40))
    a = np.clip((gy - 110) / 160, 0, 1) * np.clip((285 - gy) / 25, 0, 1)
    dx += BANGS * a * (1.4 * np.sin(w2 - gy / 80 + gx / 24 + 2.0) + 0.6 * np.sin(w3 - gx / 16 + gy / 50))
    return dx * HAND_FREE, dy * HAND_FREE

# ======================= masks / alphas =======================
def cached_mask(suffix, model):
    p = os.path.splitext(SRC)[0] + suffix
    if not os.path.exists(p):
        from rembg import new_session, remove
        from PIL import Image
        remove(Image.open(SRC).convert("RGB"), session=new_session(model), only_mask=True).save(p)
    return cv2.imread(p, 0).astype(np.float32) / 255.0

# --- head layer alpha (anime model, braids re-cut on the line art) ---
A = smooth((cached_mask("-mask.png", "isnet-anime") - 0.30) / 0.15)
cv2.rectangle(A, (370, 340), (905, 620), 1.0, -1)
_ink = cv2.morphologyEx((GRAY < 125).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
_ink = cv2.dilate(_ink, np.ones((3, 3), np.uint8)).astype(np.float32)
_zone = (poly([(525, 55), (565, 105), (550, 200), (535, 300), (485, 330), (415, 300), (398, 200), (415, 95), (465, 50)])
         | poly([(835, 55), (945, 65), (1005, 150), (1020, 255), (1005, 340), (960, 370), (895, 335), (865, 245), (845, 150)]))
_zone = (cv2.dilate(_zone, np.ones((25, 25), np.uint8)) > 0) & (gy < 340)
_near = cv2.dilate((A > 0.5).astype(np.uint8), np.ones((29, 29), np.uint8)).astype(np.float32)
A[_zone] = (_ink * _near)[_zone]
A = np.maximum(A, cv2.dilate(A, np.ones((3, 3), np.uint8)))
_inkS = ((GRAY < 110).astype(np.float32) * STRAND_ZONE) * (A > 0.5)
A_HEAD = aa(np.maximum(A * np.maximum(HEADHARD, ARM_ZONE), _inkS) > 0.5)

# --- body layer alpha: hand-traced silhouette, edge refined by a general model ---
BODY_OUT = poly([(372, 410), (392, 368), (430, 345), (480, 335), (540, 330), (900, 330), (960, 345),
                 (1010, 360), (1045, 400), (1070, 460), (1090, 510), (1150, 490), (1220, 485),
                 (1262, 500), (1265, 560), (1340, 575), (1400, 595), (1445, 600), (1462, 638),
                 (1520, 650), (1556, 640), (1556, 858), (445, 858), (442, 780), (446, 720),
                 (420, 690), (385, 645), (368, 570), (366, 480)])
_core = cv2.erode(BODY_OUT, np.ones((29, 29), np.uint8))
_gen = (cached_mask("-mask-body.png", "isnet-general-use") > 0.3).astype(np.uint8)
_body = (_core | (BODY_OUT & _gen)) & (1 - _hh)
# under the jaw / palm / braid tips the body continues beneath the head: when the head lifts,
# what is uncovered there is body (collar, cuff, shoulder frill), not background
NECK = _hh & cv2.dilate(_body, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (45, 45)))
A_BODY = aa(_body | NECK)

# --- lace skirt: light lace only (dark navy fabric stays), inside a zone under the ruffle line
LACE_ZONE = poly([(1085, 560), (1150, 498), (1215, 483), (1255, 494), (1262, 555), (1300, 565),
                  (1340, 573), (1400, 592), (1442, 595), (1460, 635), (1500, 645), (1550, 635),
                  (1550, 700), (1495, 725), (1420, 735), (1415, 848), (1090, 848), (1080, 640)])
# light lace, with its interior shadows filled in (15 px closing) so each ruffle moves as one
# piece; only large dark areas (navy fabric, background) stay out. Frame ornament excluded.
_lace = cv2.morphologyEx((GRAY > 120).astype(np.uint8), cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
_lace = cv2.morphologyEx(_lace, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)) & LACE_ZONE
A_LACE = aa(_lace)
LACE_W = smooth((gx - 1090) / 110.0)          # attached to the body on the left

# --- accessories ---
PEND_PIV = (705.0, 452.0)                     # jewelled chain hangs from the collar brooch
A_PEND = aa(poly([(660, 446), (668, 520), (681, 580), (693, 640), (700, 660), (730, 670),
                  (766, 662), (769, 640), (759, 580), (749, 520), (740, 480), (728, 446)]))
RIBBONS = [  # (zone, pivot = knot, base amplitude deg, phase)
    (poly([(728, 658), (748, 658), (762, 690), (782, 745), (778, 762), (755, 762), (735, 710), (726, 680)]),
     (737.0, 664.0), 1.4, 0.3),
    (poly([(914, 734), (940, 734), (948, 742), (962, 768), (988, 800), (962, 815), (940, 806), (925, 776),
           (912, 748)]),                    # stays clear of the red cuff band
     (930.0, 732.0), 1.8, 1.9),
]
RIB_ALPHA = [aa(cv2.dilate(((GRAY < 60) & (z > 0)).astype(np.uint8), np.ones((3, 3), np.uint8)) & z)
             for z, _, _, _ in RIBBONS]
A_ACC = np.clip(A_PEND + sum(RIB_ALPHA), 0, 1)

# --- body micro-motion weights ---
BREATH_W = (np.exp(-((gx - 720) / 420) ** 2) * smooth((800 - gy) / 300.0)).astype(np.float32)
RCUFF = soft_poly([(835, 672), (1002, 672), (1008, 808), (832, 808)], 14)
# cloth hangs from an attachment line and only its free edges swing: weight = distance from it
CUFF_EDGE = smooth(np.abs(gy - (715 + (gx - 840) * 0.125)) / 60.0)   # wrist band ~y715..735
FRILL_EDGE = smooth((gy - 752) / 70.0)                               # sleeve frills hang down
LACE_EDGE = smooth((860 - gy) / 250.0)                               # ruffle tops are free
FRILL = soft_poly([(600, 752), (1000, 752), (1000, 850), (600, 850)], 10) * smooth((852 - gy) / 12)

# ======================= plates (inpainted fills for uncovered slivers) =======================
u8 = img.astype(np.uint8)
_allA = np.maximum.reduce([A_HEAD, A_BODY, A_LACE])
BG_PLATE = cv2.inpaint(u8, ring(_allA, 4, 22) * 255, 5, cv2.INPAINT_TELEA).astype(np.float32)
_bh = ring(A_HEAD, 4, 22) | cv2.dilate(NECK, np.ones((9, 9), np.uint8)) \
      | cv2.dilate((A_ACC > 0.02).astype(np.uint8), np.ones((7, 7), np.uint8)) | ring(A_LACE, 4, 14)
BODY_SRC = cv2.inpaint(u8, _bh * 255, 5, cv2.INPAINT_TELEA).astype(np.float32)

# ======================= timelines =======================
def breath(t):                            # 4 s cycle: 40 % inhale, 60 % exhale, eased
    p = (t % 4.0) / 4.0
    return 0.5 - 0.5 * np.cos(np.pi * p / 0.4) if p < 0.4 else 0.5 + 0.5 * np.cos(np.pi * (p - 0.4) / 0.6)

BLINKS = [0.8, 4.45, 6.85, 7.2]           # the last two = a double blink
def blink_curve(t):
    c = 0.0
    for b in BLINKS:
        u = t - b
        if u < 0: continue
        if u < 0.07:   v = u / 0.07
        elif u < 0.11: v = 1.0
        elif u < 0.26: v = 1 - (u - 0.11) / 0.15
        else: continue
        c = max(c, smooth(v))
    return c

GLANCES = [(2.4, 4.2, 2.4), (5.9, 6.6, -0.9)]   # (start, end, shift px): saccades take 0.12 s
def gaze(t):
    g = 0.0
    for a, b, amp in GLANCES:
        if a <= t < b: g += amp * smooth((t - a) / 0.12) * (1 - smooth((t - (b - 0.12)) / 0.12))
    return g

# ======================= high-res buffers =======================
W2, H2 = int(round(W * K)), int(round(H * K))
R = lambda a, it=cv2.INTER_LANCZOS4: cv2.resize(a, (W2, H2), interpolation=it)
big = R(img); bigPlate = R(BG_PLATE)
# body source = original, except where an upper layer (head / lace / accessories) used to sit
_up = R(np.maximum.reduce([A_HEAD, A_LACE, A_ACC]), cv2.INTER_LINEAR)[..., None]
bigBody = _up * R(BODY_SRC) + (1 - _up) * big
bA = {n: R(a, cv2.INTER_LINEAR)[..., None] for n, a in
      dict(head=A_HEAD, body=A_BODY, lace=A_LACE, acc=A_ACC).items()}
_au = R(_allA, cv2.INTER_LINEAR)[..., None]
bigBG = _au * bigPlate + (1 - _au) * big   # static layer: original except under the character
gx2, gy2 = np.meshgrid(np.arange(W2, dtype=np.float32), np.arange(H2, dtype=np.float32))
up = lambda f: cv2.resize(f.astype(np.float32), (W2, H2), interpolation=cv2.INTER_LINEAR)

def warp(src, alpha, dx, dy):
    MX = up(dx) * K + gx2; MY = up(dy) * K + gy2
    c = cv2.remap(src, MX, MY, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    a = cv2.remap(alpha, MX, MY, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)[..., None]
    return c, a

def frame(i):
    t = i / FPS
    b = breath(t)
    # ---- body: breathing rise (chest 2.6 px, shoulders ~1.5 px) + cuff / frill flutter (cloth)
    bdy = 2.6 * b * BREATH_W
    # Cloth moves as whole pieces (no short-wavelength ripples - those read as heat haze):
    # a slow breeze sway, phase delayed only over very long distances, growing toward free edges.
    ph = (gx - 840) / 400.0
    sway = 1.4 * np.sin(TAU * t / 2.0 + 0.4 - ph) + 0.4 * np.sin(TAU * t / (4 / 3) + 1.1 - ph)
    lift = 0.6 * np.sin(TAU * t / 2.0 + 1.6 - ph)
    fdx = RCUFF * CUFF_EDGE * sway + FRILL * FRILL_EDGE * 0.6 * sway
    fdy = RCUFF * CUFF_EDGE * lift + FRILL * FRILL_EDGE * 0.4 * lift
    body_dx, body_dy = fdx, bdy + fdy
    # ---- lace skirt: fine quick ripples travelling with the breeze
    # lace skirt: each ruffle sways as a unit; a gust travels left->right over ~2000 px
    lp = (gx - 1090) / 350.0 + gy / 400.0
    lw = LACE_W * LACE_EDGE
    lace_dx = lw * (1.6 * np.sin(TAU * t / (8 / 3) - lp) + 0.5 * np.sin(TAU * t / 1.6 - 1.3 * lp + 0.9))
    lace_dy = lw * 0.8 * np.sin(TAU * t / (8 / 3) - lp + 1.3)
    # ---- accessories ride on the chest, plus their own pendulum motion
    acc_dx = np.zeros((H, W), np.float32); acc_dy = bdy.copy()
    pa = np.deg2rad(0.55 * np.sin(TAU * t / 2.0 - 0.9) + 0.25 * np.sin(TAU * t / 4.0 - 2.0))  # metal
    rx, ry = rot_field(pa, PEND_PIV)
    acc_dx += A_PEND * rx; acc_dy += A_PEND * ry
    for (z, piv, amp, ph), ra in zip(RIBBONS, RIB_ALPHA):
        r = np.hypot(gx - piv[0], gy - piv[1])
        ang = np.deg2rad(amp * np.sin(TAU * t / 1.6 + ph) + 0.4 * amp * np.sin(TAU * t / 1.0 + ph * 2))
        ang = ang * (0.6 + 0.4 * np.clip(r / 80, 0, 1.5))          # fabric bends toward the tip
        rx_ = gx - piv[0]; ry_ = gy - piv[1]
        sx = np.cos(ang) * rx_ - np.sin(ang) * ry_ - rx_; sy = np.sin(ang) * rx_ + np.cos(ang) * ry_ - ry_
        zz = cv2.dilate(z, np.ones((9, 9), np.uint8)).astype(np.float32)
        acc_dx += zz * sx; acc_dy += zz * sy
    # ---- head: rigid tiny tilt (8 s) on top of the breath, + hair
    s = 0.5 - 0.5 * np.cos(TAU * t / 8.0)
    lift, ang = 2.5 * s, np.deg2rad(0.9) * s
    hx, hy = rot_field(ang, PIV); hy = hy + lift
    hb = 2.6 * b * float(BREATH_W[450, 705])                       # breath carried at the neck
    hdx, hdy = hair_field(t)
    head_dx = HEAD * hx + hdx
    head_dy = HEAD * (hy + hb) + (1 - HEAD) * bdy + hdy
    # eyes in source space: gaze (horizontal, between lids), then blink (vertical)
    mx = gx + head_dx; my = gy + head_dy
    myA = my.copy()
    # expression: evaluated at the head-warped source position so it rides along with the head
    fdx_, fdy_ = face_field(t, b)
    m32x, m32y = mx.astype(np.float32), my.astype(np.float32)
    mx = mx + cv2.remap(fdx_, m32x, m32y, cv2.INTER_LINEAR)
    my = my + cv2.remap(fdy_, m32x, m32y, cv2.INTER_LINEAR)
    g = gaze(t)
    if g != 0:
        mx = mx - g * cv2.remap(GAZE_W, mx.astype(np.float32), my.astype(np.float32), cv2.INTER_LINEAR)
    bc = blink_curve(t)
    if bc > 0:
        my = cv2.remap(blink_map(bc), mx.astype(np.float32), my.astype(np.float32), cv2.INTER_LINEAR)
    # ---- composite back -> front
    out = bigBG.copy()
    for src, al, dx, dy in ((bigBody, bA['body'], body_dx, body_dy),
                            (big, bA['lace'], lace_dx, lace_dy),
                            (big, bA['acc'], acc_dx, acc_dy)):
        c, a = warp(src, al, dx, dy)
        out = a * c + (1 - a) * out
    MX = up(mx - gx) * K + gx2; MY = up(my - gy) * K + gy2; MYA = up(myA - gy) * K + gy2
    c = cv2.remap(big, MX, MY, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    a = cv2.remap(bA['head'], MX, MYA, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)[..., None]
    out = a * c + (1 - a) * out
    return np.clip(out, 0, 255).astype(np.uint8)

if __name__ == "__main__":
    SNAP = {int(x) for x in os.environ.get("SNAP", "").split(",") if x}
    mp4 = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                            "-s", f"{W2}x{H2}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow",
                            "-crf", "14", "-profile:v", "high", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                            "-tag:v", "avc1", os.path.join(OUT, "gothic-lolita.mp4")], stdin=subprocess.PIPE)
    wb = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                           "-s", f"{W2}x{H2}", "-r", str(FPS), "-i", "-", "-vf", "fps=24,scale=1280:-1:flags=lanczos",
                           "-c:v", "libwebp_anim", "-quality", "66", "-compression_level", "4", "-loop", "0",
                           os.path.join(OUT, "gothic-lolita.webp")], stdin=subprocess.PIPE)
    with Pool(WORKERS) as pool:
        for i, fr in enumerate(pool.imap(frame, range(N), chunksize=4)):
            mp4.stdin.write(fr.tobytes())
            wb.stdin.write(fr.tobytes())
            if i in SNAP: cv2.imwrite(os.path.join(OUT, f"snap{i:03d}.png"), fr)
            if i % 60 == 0: print("frame", i, flush=True)
    for p in (mp4, wb): p.stdin.close(); p.wait()
    print("done", N, W2, H2)
