#!/usr/bin/env python3
"""Rotate each object in place to match its dashed target, then slide it right into the target."""
import os
import subprocess
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
N_FRAMES = 48
FPS = 16
SPLIT_X = 485          # blank column band between left objects and right targets
STROKE = np.array([120, 120, 120], dtype=np.float32)


def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def main():
    img = cv2.imread(FIRST)[:, :, ::-1].copy()          # RGB uint8
    H, W = img.shape[:2]
    imf = img.astype(np.float32)
    nonwhite = img.min(axis=2) < 250

    # ---- dashed targets: distance field to nearest dash pixel -------------------------------
    dash = nonwhite.copy()
    dash[:, :SPLIT_X] = False
    dist = cv2.distanceTransform((~dash).astype(np.uint8), cv2.DIST_L2, 5)

    # ---- objects ------------------------------------------------------------------------------
    left = nonwhite.copy()
    left[:, SPLIT_X:] = False
    n, lab, stats, cents = cv2.connectedComponentsWithStats(left.astype(np.uint8), connectivity=8)

    # background: every object pixel (plus faint anti-aliasing halo) turned white
    halo = cv2.dilate(left.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    faint = img.min(axis=2) < 255
    background = img.copy()
    background[halo & faint] = 255

    objects = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 200:
            continue
        comp = (lab == i)
        comp = cv2.dilate(comp.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & faint
        # remove any pixels that belong to the dilated halo of other components (none expected)
        ys, xs = np.nonzero(comp)
        # centroid of the filled shape (area centroid of the solid mask)
        cx, cy = xs.mean(), ys.mean()

        # RGBA sprite: outer ring = anti-aliased stroke over white -> recover alpha
        rgba = np.zeros((H, W, 4), np.float32)
        rgba[comp, :3] = imf[comp]
        rgba[comp, 3] = 1.0
        inner = cv2.erode(comp.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        ring = comp & ~inner
        a = np.clip((255.0 - imf[ring].mean(axis=1)) / (255.0 - STROKE[0]), 0, 1)
        rgba[ring, :3] = STROKE
        rgba[ring, 3] = a
        # premultiply
        rgba[..., :3] *= rgba[..., 3:4]

        # contour of the stroke centre line (~1.5px inside the outer edge) for fitting
        er = cv2.erode(comp.astype(np.uint8), np.ones((3, 3), np.uint8))
        cnts, _ = cv2.findContours(er, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
        pts = cnt - np.array([cx, cy])

        # is it a circle? (contour radius nearly constant)
        r = np.hypot(pts[:, 0], pts[:, 1])
        is_circle = (r.std() / r.mean()) < 0.03

        def score(theta_deg, dx, dy=0.0):
            th = np.deg2rad(theta_deg)
            c, s = np.cos(th), np.sin(th)
            px = c * pts[:, 0] - s * pts[:, 1] + cx + dx
            py = s * pts[:, 0] + c * pts[:, 1] + cy + dy
            xi = np.clip(np.round(px).astype(int), 0, W - 1)
            yi = np.clip(np.round(py).astype(int), 0, H - 1)
            return np.minimum(dist[yi, xi], 25.0).mean()

        # coarse search
        thetas = [0.0] if is_circle else np.arange(-180, 180, 2.0)
        dxs = np.arange(380, 600, 2.0)
        best = (1e9, 0.0, 0.0)
        for th in thetas:
            for dx in dxs:
                sc = score(th, dx)
                if sc < best[0]:
                    best = (sc, th, dx)
        # fine search
        _, th0, dx0 = best
        for step in (0.5, 0.1):
            ths = [th0] if is_circle else np.arange(th0 - 4 * step * 2, th0 + 4 * step * 2 + 1e-9, step)
            for th in ths:
                for dx in np.arange(dx0 - 4 * step * 2, dx0 + 4 * step * 2 + 1e-9, step):
                    sc = score(th, dx)
                    if sc < best[0]:
                        best = (sc, th, dx)
            _, th0, dx0 = best
        sc, theta, dx = best
        # choose the smallest equivalent rotation for symmetric shapes
        if not is_circle:
            cands = []
            for k in range(3, 9):
                period = 360.0 / k
                t2 = (theta + period / 2) % period - period / 2
                if abs(score(t2, dx) - sc) < 0.05:
                    cands.append(t2)
            if cands:
                theta = min(cands, key=abs)
        print(f"object {i}: centre=({cx:.1f},{cy:.1f}) circle={is_circle} "
              f"theta={theta:.2f} deg dx={dx:.2f} score={sc:.3f}")
        objects.append(dict(rgba=rgba, cx=cx, cy=cy, theta=theta, dx=dx))

    # ---- render -------------------------------------------------------------------------------
    os.makedirs(OUT_DIR, exist_ok=True)
    frames = []
    for f in range(N_FRAMES):
        t = f / (N_FRAMES - 1)
        rot = smoothstep(t / 0.5)                 # first half: rotate in place
        mov = smoothstep((t - 0.5) / 0.5)         # second half: slide right
        if f == 0:
            frames.append(img.copy())
            continue
        canvas = background.astype(np.float32)
        for ob in objects:
            th = ob["theta"] * rot
            dx = ob["dx"] * mov
            M = cv2.getRotationMatrix2D((ob["cx"], ob["cy"]), -th, 1.0)  # image y is down
            M[0, 2] += dx
            warped = cv2.warpAffine(ob["rgba"], M, (W, H), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            a = np.clip(warped[..., 3:4], 0, 1)
            canvas = canvas * (1 - a) + np.clip(warped[..., :3], 0, 255)
        frames.append(np.clip(np.round(canvas), 0, 255).astype(np.uint8))

    # write with ffmpeg (H.264, yuv420p)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
           "-movflags", "+faststart", OUT]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fr in frames:
        p.stdin.write(np.ascontiguousarray(fr).tobytes())
    p.stdin.close()
    p.wait()
    if p.returncode != 0:
        raise SystemExit("ffmpeg failed")
    cv2.imwrite(os.path.join(OUT_DIR, "last_frame.png"), frames[-1][:, :, ::-1])
    cv2.imwrite(os.path.join(OUT_DIR, "mid_frame.png"), frames[N_FRAMES // 2][:, :, ::-1])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
