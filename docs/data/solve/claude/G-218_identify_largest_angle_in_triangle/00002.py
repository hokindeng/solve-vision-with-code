#!/usr/bin/env python3
"""Regenerate /app/output/video.mp4: circle the vertex with the largest interior angle."""
import math, os, subprocess
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

BASE = "/app/first_frame.png"
OUT = "/app/output/video.mp4"
FPS, N_FRAMES, SS = 16, 22, 4  # SS = supersampling factor for anti-aliased overlays


def find_vertices(img_gray):
    mask = (img_gray < 128).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)
    ap = cv2.approxPolyDP(c, 4, True).reshape(-1, 2)
    while len(ap) > 3:  # keep the three most-spread corners
        ap = cv2.approxPolyDP(c, 4 + len(ap), True).reshape(-1, 2)
    # refine each corner to the centroid of dark pixels within a small window
    pts = []
    for x, y in ap:
        win = mask[max(0, y - 4):y + 5, max(0, x - 4):x + 5]
        ys, xs = np.nonzero(win)
        pts.append((max(0, x - 4) + xs.mean(), max(0, y - 4) + ys.mean()))
    return [np.array(p, float) for p in pts]


def angle_at(p, q, r):
    v1, v2 = q - p, r - p
    return math.degrees(math.acos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))))


def ease(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def main():
    base = Image.open(BASE).convert("RGB")
    W, H = base.size
    V = find_vertices(np.array(base.convert("L")))
    angs = [angle_at(V[i], V[(i + 1) % 3], V[(i + 2) % 3]) for i in range(3)]
    big = int(np.argmax(angs))
    print("vertices:", [tuple(np.round(v, 1)) for v in V])
    print("angles:", [round(a, 1) for a in angs], "-> largest at vertex", big)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 26 * SS)
    except Exception:
        font = ImageFont.load_default()

    frames = []
    for f in range(N_FRAMES):
        ov = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
        d = ImageDraw.Draw(ov)
        # --- step 1 (frames 1..9): measure each angle in turn (arc + label) ---
        for i in range(3):
            t0 = 1 + 3 * i
            a = ease((f - t0 + 1) / 3.0)
            # fade-out of annotations during the final circle phase
            fade = 1.0 - ease((f - 15) / 5.0)
            alpha = int(255 * a * fade)
            if alpha <= 0:
                continue
            p, q, r = V[i], V[(i + 1) % 3], V[(i + 2) % 3]
            a1 = math.degrees(math.atan2(q[1] - p[1], q[0] - p[0]))
            a2 = math.degrees(math.atan2(r[1] - p[1], r[0] - p[0]))
            # choose the sweep that is the interior angle
            sweep = (a2 - a1) % 360
            if sweep > 180:
                a1, a2 = a2, a1
                sweep = 360 - sweep
            highlight = i == big and f >= 11
            hl = ease((f - 10) / 3.0) if highlight else 0.0
            col = (int(30 + (220 - 30) * hl), int(90 - 60 * hl), int(200 - 160 * hl), alpha)
            rad = (34 + 8 * hl) * SS
            width = int((3 + 2 * hl) * SS)
            bbox = [p[0] * SS - rad, p[1] * SS - rad, p[0] * SS + rad, p[1] * SS + rad]
            d.arc(bbox, a1, a1 + sweep * a, fill=col, width=width)
            if a >= 1.0:
                mid = math.radians(a1 + sweep / 2)
                lx = p[0] * SS + math.cos(mid) * rad * 2.1
                ly = p[1] * SS + math.sin(mid) * rad * 2.1
                d.text((lx, ly), f"{angs[i]:.0f}°", fill=col, font=font, anchor="mm")
        # --- step 3 (frames 13..21): draw the red circle around the largest-angle vertex ---
        c = ease((f - 12) / 9.0)
        if c > 0:
            p = V[big]
            R = 42 * SS
            bbox = [p[0] * SS - R, p[1] * SS - R, p[0] * SS + R, p[1] * SS + R]
            d.arc(bbox, -90, -90 + 360 * c, fill=(230, 20, 20, 255), width=5 * SS)
        ov = ov.resize((W, H), Image.LANCZOS)
        frame = Image.alpha_composite(base.convert("RGBA"), ov).convert("RGB")
        frames.append(np.array(frame))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-crf", "12", "-preset", "slow", OUT]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fr in frames:
        proc.stdin.write(fr.tobytes())
    proc.stdin.close()
    proc.wait()
    Image.fromarray(frames[-1]).save("/app/output/last_frame.png")
    print("wrote", OUT, len(frames), "frames")


if __name__ == "__main__":
    main()
