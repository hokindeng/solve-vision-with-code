#!/usr/bin/env python3
"""Two balls move toward each other at equal speed and merge at the midpoint.
Overlapping regions use additive light mixing (channel-wise sum, clipped)."""
import subprocess
import numpy as np
from PIL import Image, ImageDraw

W = H = 1024
FPS = 16
N_FRAMES = 80
FIRST = "/app/first_frame.png"
OUT = "/app/output/video.mp4"

orig = np.array(Image.open(FIRST).convert("RGB"))

# --- detect the two balls (solid fills, non-white, non-black) -------------
flat = orig.reshape(-1, 3)
cols, counts = np.unique(flat, axis=0, return_counts=True)
order = np.argsort(-counts)
balls = []
for i in order:
    c = tuple(int(v) for v in cols[i])
    if c in [(255, 255, 255), (0, 0, 0)]:
        continue
    m = np.all(orig == c, axis=2)
    ys, xs = np.nonzero(m)
    balls.append(dict(color=c, cx=xs.mean(), cy=ys.mean(), r_fill=(xs.max() - xs.min() + 1) / 2))
    if len(balls) == 2:
        break
assert len(balls) == 2, "expected two coloured balls"
R = 120          # outer radius (fill radius 118 + 2px outline), verified to match first frame
OUTLINE_W = 2
OUTLINE = (0, 0, 0)

# background = first frame with both balls erased (outline included)
bg = orig.copy()
yy, xx = np.mgrid[0:H, 0:W]
for b in balls:
    erase = (xx - b["cx"]) ** 2 + (yy - b["cy"]) ** 2 <= (R + 1.5) ** 2
    bg[erase] = 255  # background is uniform white behind the balls

mid = np.array([(balls[0]["cx"] + balls[1]["cx"]) / 2, (balls[0]["cy"] + balls[1]["cy"]) / 2])
starts = [np.array([b["cx"], b["cy"]]) for b in balls]


def ease(t):  # smooth start/stop, same profile for both balls -> equal speeds
    return t * t * (3 - 2 * t)


def draw_ball_layers(center, color):
    """Return (fill_mask, outline_mask) drawn with PIL exactly like the source image."""
    cx, cy = center
    box = [cx - R, cy - R, cx + R, cy + R]
    fill_im = Image.new("L", (W, H), 0)
    ImageDraw.Draw(fill_im).ellipse(box, fill=255, outline=0, width=OUTLINE_W)
    ring_im = Image.new("L", (W, H), 0)
    ImageDraw.Draw(ring_im).ellipse(box, fill=0, outline=255, width=OUTLINE_W)
    return np.array(fill_im) > 0, np.array(ring_im) > 0


def render(t):
    if t <= 0:
        return orig.copy()
    p = ease(t)
    frame = bg.astype(np.int32)
    light = np.zeros((H, W, 3), np.int32)
    covered = np.zeros((H, W), bool)
    rings = np.zeros((H, W), bool)
    for b, s in zip(balls, starts):
        c = s + (mid - s) * p
        if t >= 1:
            c = mid
        fill, ring = draw_ball_layers((float(c[0]), float(c[1])), b["color"])
        light[fill] += np.array(b["color"], np.int32)
        covered |= fill
        rings |= ring
    # additive light mixing inside the beams, clipped to displayable range
    frame[covered] = np.clip(light[covered], 0, 255)
    # outlines: keep each ball's edge, but not where it passes through the other ball's light
    rings &= ~covered
    frame[rings] = OUTLINE
    return frame.astype(np.uint8)


def main():
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
           "-movflags", "+faststart", OUT]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(N_FRAMES):
        t = i / (N_FRAMES - 1)
        f = render(t)
        proc.stdin.write(f.tobytes())
    proc.stdin.close()
    proc.wait()
    assert proc.returncode == 0, "ffmpeg failed"
    print("wrote", OUT)


if __name__ == "__main__":
    main()
