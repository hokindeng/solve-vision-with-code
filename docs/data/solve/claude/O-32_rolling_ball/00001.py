#!/usr/bin/env python3
"""Animate the ball rolling / hopping along the platforms of first_frame.png."""
import subprocess, os
import numpy as np
import cv2
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
FPS, N_FRAMES = 16, 64

WHITE = (255, 255, 255)
BALL_FILL, BALL_EDGE = (255, 69, 0), (175, 0, 0)
PLAT_FILL, PLAT_EDGE = (138, 43, 226), (78, 0, 166)
SHIFT = 4  # sub-pixel bits for cv2 drawing
S = 1 << SHIFT


def analyze(img):
    """Locate ball (centre, radius) and platform centres from the first frame."""
    r, g, b = [img[..., i].astype(int) for i in range(3)]
    purple = (b > 120) & (r < 200) & (g < 120)
    nonwhite = ~((r > 245) & (g > 245) & (b > 245))
    ball = nonwhite & ~purple
    ys, xs = np.nonzero(ball)
    bc = ((xs.min() + xs.max()) / 2.0, (ys.min() + ys.max()) / 2.0)
    br = (xs.max() - xs.min() + 1) / 2.0
    n, lab, stats, cents = cv2.connectedComponentsWithStats(purple.astype(np.uint8))
    plats = []
    for i in range(1, n):
        pts = np.column_stack(np.nonzero(lab == i))[:, ::-1].astype(np.float32)
        (cx, cy), (w, h), ang = cv2.minAreaRect(pts)
        plats.append(dict(c=(cx, cy), size=(w + h) / 2, ang=ang, mask=lab == i, n=stats[i, 4]))
    return bc, br, plats, purple


def draw_square(canvas, c, side, ang_deg, fill, edge, edge_w=2):
    box = cv2.boxPoints(((c[0], c[1]), (side, side), ang_deg))
    pts = np.round(box * S).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(canvas, [pts], edge, lineType=cv2.LINE_8, shift=SHIFT)
    inner = cv2.boxPoints(((c[0], c[1]), (side - 2 * edge_w, side - 2 * edge_w), ang_deg))
    pts = np.round(inner * S).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(canvas, [pts], fill, lineType=cv2.LINE_8, shift=SHIFT)


def draw_ball(canvas, c, r, edge_w=2):
    cc = (int(round(c[0] * S)), int(round(c[1] * S)))
    cv2.circle(canvas, cc, int(round(r * S)), BALL_EDGE, -1, cv2.LINE_8, SHIFT)
    cv2.circle(canvas, cc, int(round((r - edge_w) * S)), BALL_FILL, -1, cv2.LINE_8, SHIFT)


def fit_occluded_platform(plat, ball_mask, ref_size):
    """Platform partially hidden under the ball: fit centre/angle to visible pixels."""
    vis = plat["mask"]
    valid = ~ball_mask
    cx0, cy0 = plat["c"]
    best = None
    h, w = vis.shape
    for ang in np.arange(plat["ang"] - 15, plat["ang"] + 15.1, 1.0):
        for dx in np.arange(-4, 4.1, 0.5):
            for dy in np.arange(-4, 4.1, 0.5):
                cnv = np.zeros((h, w, 3), np.uint8)
                draw_square(cnv, (cx0 + dx, cy0 + dy), ref_size, ang, PLAT_FILL, PLAT_EDGE)
                m = cnv[..., 2] > 0
                inter = (m & vis & valid).sum()
                union = ((m | vis) & valid).sum()
                iou = inter / max(union, 1)
                if best is None or iou > best[0]:
                    best = (iou, cx0 + dx, cy0 + dy, ang)
    return best


def smoothstep(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def main():
    first = np.array(Image.open(FIRST).convert("RGB"))
    bc, br, plats, purple = analyze(first)
    # order platforms along the path, starting nearest the ball
    plats.sort(key=lambda p: np.hypot(p["c"][0] - bc[0], p["c"][1] - bc[1]))
    ref_size = np.median([p["size"] for p in plats[1:]])

    # background: first frame with the ball removed
    yy, xx = np.mgrid[0:first.shape[0], 0:first.shape[1]]
    ball_mask = (xx - bc[0]) ** 2 + (yy - bc[1]) ** 2 <= (br + 1.5) ** 2
    bg = first.copy()
    bg[ball_mask] = WHITE
    # rebuild the platform hidden under the ball
    p0 = plats[0]
    if p0["n"] < 0.97 * np.median([p["n"] for p in plats[1:]]):
        iou, cx, cy, ang = fit_occluded_platform(p0, ball_mask, ref_size)
        cnv = bg.copy()
        draw_square(cnv, (cx, cy), ref_size, ang, PLAT_FILL, PLAT_EDGE)
        bg[ball_mask] = cnv[ball_mask]
        p0["c"] = (cx, cy)

    # --- motion: waypoints = ball start, then each platform centre in order
    way = [bc] + [p["c"] for p in plats]
    n_seg = len(way) - 1
    move_frames, rest_frames = 7, 2
    per_seg = move_frames + rest_frames
    # timeline for frames 1..N-1
    positions = [bc]
    for f in range(1, N_FRAMES):
        seg = min((f - 1) // per_seg, n_seg - 1)
        k = (f - 1) - seg * per_seg
        a, b = np.array(way[seg]), np.array(way[seg + 1])
        if k < move_frames:
            t = (k + 1) / move_frames
            s = smoothstep(t)
            pos = a + (b - a) * s
            hop = 14.0 * np.sin(np.pi * t)  # small arc between platforms
            pos = pos + np.array([0.0, -hop])
        else:
            pos = b
        positions.append(tuple(pos))
    positions[-1] = way[-1]  # come to rest on the final platform

    frames = []
    for f in range(N_FRAMES):
        if f == 0:
            frames.append(first.copy())
            continue
        img = bg.copy()
        draw_ball(img, positions[f], br)
        frames.append(img)

    os.makedirs(OUT_DIR, exist_ok=True)
    raw = os.path.join(OUT_DIR, "_frames.raw")
    with open(raw, "wb") as fh:
        for fr in frames:
            fh.write(np.ascontiguousarray(fr).tobytes())
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", "1024x1024", "-r", str(FPS), "-i", raw,
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
           "-r", str(FPS), OUT]
    subprocess.run(cmd, check=True)
    os.remove(raw)
    Image.fromarray(frames[-1]).save(os.path.join(OUT_DIR, "last_frame.png"))
    print("wrote", OUT, len(frames), "frames")


if __name__ == "__main__":
    main()
