#!/usr/bin/env python3
"""Black ball eats the colored balls in the only valid order (smallest first, growing after each)."""
import subprocess, numpy as np
from PIL import Image, ImageDraw

SRC = "/app/first_frame.png"
OUT = "/app/output/video.mp4"
FPS, N = 16, 108
GROW = 0.75  # radius gained = GROW * eaten radius

base = np.array(Image.open(SRC).convert("RGB"))
H, W = base.shape[:2]

def find(color):
    m = np.all(base == np.array(color, np.uint8), axis=2)
    ys, xs = np.nonzero(m)
    return m, (int(round(xs.mean())), int(round(ys.mean()))), int(round((xs.max() - xs.min()) / 2))

black_mask, black_c, black_r = find((0, 0, 0))
balls = {}
for name, col in [("green", (60, 179, 113)), ("blue", (70, 130, 180)),
                  ("yellow", (255, 215, 0)), ("orange", (255, 140, 0))]:
    m, c, r = find(col)
    balls[name] = dict(color=col, mask=m, c=c, r=r)

# Solve: greedily eat the largest ball that is still smaller than us; fall back to sorted order.
order, r, remaining = [], black_r, dict(balls)
while remaining:
    cands = [k for k, b in remaining.items() if b["r"] < r]
    assert cands, "no edible ball; puzzle unsolvable with this growth rule"
    k = min(cands, key=lambda k: remaining[k]["r"])  # smallest first: safest
    order.append(k); r += GROW * remaining[k]["r"]; remaining.pop(k)
print("eating order:", order)

# Timeline: each step = travel + eat; final hold.
steps = len(order)
HOLD = 6
per = (N - HOLD) // steps                # 25 frames per step
EAT = 6
def ease(t): return t * t * (3 - 2 * t)  # smoothstep

# Background with all colored balls removed (pure white), then re-add the ones still present.
bg_clean = base.copy()
bg_clean[black_mask] = 255
for b in balls.values(): bg_clean[b["mask"]] = 255

frames = []
pos, rad = np.array(black_c, float), float(black_r)
eaten = []
for f in range(N):
    step = min(f // per, steps - 1)
    k = order[step]
    t_in = f - step * per
    target = balls[k]
    start_pos = np.array(black_c, float) if step == 0 else np.array(balls[order[step - 1]]["c"], float)
    start_r = black_r + GROW * sum(balls[o]["r"] for o in order[:step])
    end_r = start_r + GROW * target["r"]
    travel = per - EAT
    img = bg_clean.copy()
    if f >= steps * per:  # hold at the end
        cur, cr, shrink = np.array(target["c"], float), end_r, 0.0
        present = []
    elif t_in < travel:
        u = ease(t_in / travel)
        cur, cr, shrink = start_pos + (np.array(target["c"]) - start_pos) * u, start_r, 1.0
        present = order[step:]
    else:
        u = (t_in - travel + 1) / EAT
        cur, cr, shrink = np.array(target["c"], float), start_r + (end_r - start_r) * u, 1.0 - u
        present = order[step + 1:]
    pil = Image.fromarray(img); d = ImageDraw.Draw(pil)
    for o in present:  # untouched balls: exact original pixels
        arr = np.array(pil); arr[balls[o]["mask"]] = base[balls[o]["mask"]]; pil = Image.fromarray(arr); d = ImageDraw.Draw(pil)
    if shrink > 0 and t_in >= travel and f < steps * per:  # shrinking target
        sr = target["r"] * shrink; cx, cy = target["c"]
        if sr >= 0.5: d.ellipse([cx - sr, cy - sr, cx + sr, cy + sr], fill=target["color"])
    x, y = cur
    d.ellipse([x - cr, y - cr, x + cr, y + cr], fill=(0, 0, 0))
    frames.append(np.array(pil))

assert np.array_equal(frames[0], base), "frame 0 must equal first_frame.png"
p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                      "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                      "-crf", "18", OUT], stdin=subprocess.PIPE)
for fr in frames: p.stdin.write(fr.tobytes())
p.stdin.close(); p.wait()
print("wrote", OUT, len(frames), "frames")
