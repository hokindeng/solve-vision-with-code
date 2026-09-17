#!/usr/bin/env python3
"""Identify the vertex with the largest interior angle of the triangle in
first_frame.png and circle it in red, animated step by step."""
import os, subprocess, math
import numpy as np, cv2
from PIL import Image, ImageDraw

APP = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(APP, "first_frame.png")
OUT_DIR = os.path.join(APP, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
N, FPS = 22, 16


def find_triangle(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = (gray < 128).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea)
    eps = 2.0
    while True:
        ap = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
        if len(ap) <= 3:
            break
        eps += 1.0
    pts = ap.astype(float)
    # refine: fit each edge with a line using the stroke pixels near it, intersect adjacent lines
    ys, xs = np.nonzero(mask)
    P = np.stack([xs, ys], 1).astype(float)
    lines = []
    for i in range(3):
        a, b = pts[i], pts[(i + 1) % 3]
        d = b - a; L = np.linalg.norm(d); d /= L
        n = np.array([-d[1], d[0]])
        rel = P - a
        t = rel @ d; dist = np.abs(rel @ n)
        sel = (dist < 4) & (t > 0.15 * L) & (t < 0.85 * L)
        Q = P[sel]
        vx, vy, x0, y0 = cv2.fitLine(Q.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        lines.append((np.array([x0, y0]), np.array([vx, vy])))
    V = []
    for i in range(3):
        (p1, d1), (p2, d2) = lines[(i - 1) % 3], lines[i]
        A = np.array([d1, -d2]).T
        s = np.linalg.solve(A, p2 - p1)
        V.append(p1 + s[0] * d1)
    return np.array(V)


def angles(V):
    out = []
    for i in range(3):
        a = V[(i - 1) % 3] - V[i]; b = V[(i + 1) % 3] - V[i]
        out.append(math.degrees(math.acos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))))
    return out


def ease(t):
    return 0.5 - 0.5 * math.cos(math.pi * min(max(t, 0.0), 1.0))


def main():
    base = np.array(Image.open(SRC).convert("RGB"))
    V = find_triangle(base)
    ang = angles(V)
    k = int(np.argmax(ang))
    print("vertices:", V.round(1).tolist(), "angles:", [round(a, 1) for a in ang], "largest at", k)

    cx, cy = V[k]
    R = 28
    SS = 4  # supersampling for smooth anti-aliased strokes
    W, H = base.shape[1], base.shape[0]

    frames = []
    for f in range(N):
        canvas = Image.fromarray(base).convert("RGBA")
        over = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
        dr = ImageDraw.Draw(over)

        # Step 1 (frames 2..11): measure each interior angle in turn with a small arc.
        # Step 2 (frames 11..21): arcs fade out while a red circle sweeps around the winner.
        for i in range(3):
            t0 = 2 + 3 * i
            prog = ease((f - t0) / 3.0)
            fade = 1.0 - ease((f - 12) / 5.0)
            alpha = int(255 * prog * fade)
            if alpha <= 0:
                continue
            a = V[(i - 1) % 3] - V[i]; b = V[(i + 1) % 3] - V[i]
            a0 = math.degrees(math.atan2(a[1], a[0])); a1 = math.degrees(math.atan2(b[1], b[0]))
            d = (a1 - a0) % 360
            if d > 180:
                a0, a1 = a1, a0; d = 360 - d
            r = 34 * SS
            x, y = V[i] * SS
            col = (220, 60, 40, alpha) if i == k else (40, 110, 220, alpha)
            dr.arc([x - r, y - r, x + r, y + r], a0, a0 + d * prog, fill=col, width=2 * SS)

        sweep = ease((f - 11) / 10.0)
        if sweep > 0:
            r = R * SS
            x, y = cx * SS, cy * SS
            if sweep >= 0.999:
                dr.ellipse([x - r, y - r, x + r, y + r], outline=(255, 0, 0, 255), width=3 * SS)
            else:
                dr.arc([x - r, y - r, x + r, y + r], -90, -90 + 360 * sweep,
                       fill=(255, 0, 0, 255), width=3 * SS)

        over = over.resize((W, H), Image.LANCZOS)
        canvas.alpha_composite(over)
        frames.append(np.array(canvas.convert("RGB")))

    frames[0] = base.copy()  # first frame identical to the source
    os.makedirs(OUT_DIR, exist_ok=True)
    tmp = os.path.join(OUT_DIR, "frames")
    os.makedirs(tmp, exist_ok=True)
    for i, fr in enumerate(frames):
        Image.fromarray(fr).save(os.path.join(tmp, f"{i:03d}.png"))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                    "-i", os.path.join(tmp, "%03d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "12", "-preset", "slow", OUT], check=True)
    for fn in os.listdir(tmp):
        os.remove(os.path.join(tmp, fn))
    os.rmdir(tmp)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
