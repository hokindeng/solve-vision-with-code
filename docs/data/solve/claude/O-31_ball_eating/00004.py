#!/usr/bin/env python3
"""Black ball eats all colored balls, smallest first, growing after each meal.

Frames start from first_frame.png; only the black ball and the ball currently
being eaten are repainted, everything else is copied verbatim.
"""
import os, subprocess, shutil
import numpy as np, cv2
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
W = H = 1024
FPS = 16
N_FRAMES = 108
GROWTH = 3.0  # r_new^2 = r^2 + GROWTH * r_eaten^2


def detect_balls(img):
    mask = (np.abs(img.astype(int) - 255).sum(2) > 30).astype(np.uint8)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask)
    balls = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        col = np.median(img[lab == i].reshape(-1, 3), axis=0)
        balls.append(dict(c=np.array(cent[i], float), r=float(np.sqrt(area / np.pi)),
                          col=tuple(int(v) for v in col), bbox=(x, y, w, h)))
    return balls


def ease(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def draw_disc(img, c, r, col):
    if r <= 0.05:
        return
    sh = 8
    cv2.circle(img, (int(round(c[0] * 2**sh)), int(round(c[1] * 2**sh))),
               int(round(r * 2**sh)), col, -1, cv2.LINE_AA, sh)


def erase_disc(img, c, r):
    cv2.circle(img, (int(round(c[0] * 256)), int(round(c[1] * 256))),
               int(round((r + 2.5) * 256)), (255, 255, 255), -1, cv2.LINE_AA, 8)


def clamp_center(c, r):
    return np.array([min(max(c[0], r), W - r), min(max(c[1], r), H - r)])


def main():
    base = np.array(Image.open(SRC).convert("RGB"))
    balls = detect_balls(base)
    black = [b for b in balls if sum(b["col"]) < 100][0]
    colored = sorted([b for b in balls if sum(b["col"]) >= 100], key=lambda b: b["r"])

    # Plan: for each target, travel phase then absorb phase.
    hold_end = 3
    per = (N_FRAMES - 1 - hold_end) // len(colored)  # frames per meal
    travel_frac = 0.6

    # Precompute keyframes of the black ball state per meal.
    plan = []
    pos, rad = black["c"].copy(), black["r"]
    for i, t in enumerate(colored):
        assert rad > t["r"], "cannot eat a bigger ball"
        d = t["c"] - pos
        dist = np.linalg.norm(d)
        u = d / dist if dist > 0 else np.zeros(2)
        stop = t["c"] - u * rad          # target centre sits on black's rim
        stop = clamp_center(stop, rad)
        new_r = float(np.sqrt(rad**2 + GROWTH * t["r"] ** 2))
        end_pos = clamp_center(stop, new_r)
        plan.append(dict(start_pos=pos.copy(), start_r=rad, stop=stop, end_pos=end_pos,
                         new_r=new_r, target=t))
        pos, rad = end_pos, new_r

    os.makedirs(OUT_DIR, exist_ok=True)
    tmp = os.path.join(OUT_DIR, "_frames")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)

    for f in range(N_FRAMES):
        if f == 0:
            frame = base.copy()
        else:
            k = min((f - 1) // per, len(plan) - 1)
            local = f - 1 - k * per
            if k == len(plan) - 1 and local >= per:
                local = per  # hold at the end
            p = plan[k]
            n_travel = int(round(per * travel_frac))
            n_absorb = per - n_travel
            frame = base.copy()
            # erase black and all eaten / in-progress balls
            erase_disc(frame, black["c"], black["r"])
            for j in range(k + 1):
                erase_disc(frame, plan[j]["target"]["c"], plan[j]["target"]["r"])
            t = p["target"]
            if local <= n_travel:
                s = ease(local / n_travel) if n_travel else 1.0
                bc = p["start_pos"] + (p["stop"] - p["start_pos"]) * s
                br = p["start_r"]
                draw_disc(frame, t["c"], t["r"], t["col"])
            else:
                s = ease((local - n_travel) / max(n_absorb, 1))
                bc = p["stop"] + (p["end_pos"] - p["stop"]) * s
                br = p["start_r"] + (p["new_r"] - p["start_r"]) * s
                tc = t["c"] + (bc - t["c"]) * s
                tr = t["r"] * (1 - s)
                draw_disc(frame, tc, tr, t["col"])
            draw_disc(frame, bc, br, (0, 0, 0))
        Image.fromarray(frame).save(os.path.join(tmp, f"{f:04d}.png"))

    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                    "-i", os.path.join(tmp, "%04d.png"), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "12", "-r", str(FPS), OUT], check=True)
    shutil.rmtree(tmp)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
