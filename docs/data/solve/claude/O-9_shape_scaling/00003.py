#!/usr/bin/env python3
"""Generate the analogy-completion video: A:B :: C:? with a scaling rule.

A (T-shape, 150 px) -> B (T-shape, 126 px)  => factor 0.84.
C (arrow, 150 px) -> ? must become the arrow scaled by 0.84, centred where '?' is.
"""
import subprocess
import numpy as np
from PIL import Image, ImageDraw

ROOT = "/app"
FIRST = f"{ROOT}/first_frame.png"
OUT = f"{ROOT}/output/video.mp4"
W = H = 1024
FPS = 16
N_FRAMES = 60

FILL = (255, 0, 128)
OUTLINE = (0, 0, 0)
BG = (255, 255, 255)

# Measured from first_frame.png
A_W, B_W = 150.0, 126.0
SCALE = B_W / A_W                      # 0.84
C_CENTER = (255.0, 768.0)
Q_CENTER = (769.0, 768.0)              # centre of the '?' glyph -> target position
Q_BOX = (740, 730, 800, 806)           # region containing the '?' glyph (x0,y0,x1,y1)

# Arrow polygon in local coords (centre at origin), full size (150 x 150)
ARROW = [(-75, -37.5), (0, -37.5), (0, -75), (75, 0), (0, 75), (0, 37.5), (-75, 37.5)]


def ease(t):
    """smoothstep easing on [0,1]"""
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def render_arrow(base, center, scale, alpha=1.0, ss=4):
    """Composite the arrow (scaled) onto `base` (np uint8 HxWx3) with given alpha."""
    cx, cy = center
    pts = [(cx + x * scale, cy + y * scale) for x, y in ARROW]
    # bounding box with margin
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    x0, y0 = int(np.floor(min(xs))) - 3, int(np.floor(min(ys))) - 3
    x1, y1 = int(np.ceil(max(xs))) + 3, int(np.ceil(max(ys))) + 3
    bw, bh = x1 - x0, y1 - y0
    layer = Image.new("RGBA", (bw * ss, bh * ss), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    spts = [((x - x0) * ss, (y - y0) * ss) for x, y in pts]
    d.polygon(spts, fill=FILL + (255,), outline=OUTLINE + (255,), width=ss)
    layer = layer.resize((bw, bh), Image.LANCZOS)
    lay = np.asarray(layer).astype(np.float32)
    a = lay[..., 3:4] / 255.0 * alpha
    region = base[y0:y1, x0:x1].astype(np.float32)
    region = region * (1 - a) + lay[..., :3] * a
    base[y0:y1, x0:x1] = np.clip(region + 0.5, 0, 255).astype(np.uint8)
    return base


def main():
    first = np.asarray(Image.open(FIRST).convert("RGB")).astype(np.uint8)
    # background with '?' removed
    no_q = first.copy()
    x0, y0, x1, y1 = Q_BOX
    no_q[y0:y1, x0:x1] = BG

    # timeline (frame indices)
    F_QFADE0, F_QFADE1 = 1, 13      # '?' fades out
    F_IN0, F_IN1 = 12, 20           # full-size arrow fades in at target
    F_SC0, F_SC1 = 22, 52           # arrow shrinks 1.0 -> SCALE
    # 52..59 hold final result

    frames = []
    for i in range(N_FRAMES):
        if i == 0:
            frames.append(first.copy())
            continue
        # background: blend '?' out
        qa = 1.0 - ease((i - F_QFADE0) / (F_QFADE1 - F_QFADE0))
        img = (no_q.astype(np.float32) * (1 - qa) + first.astype(np.float32) * qa)
        img = np.clip(img + 0.5, 0, 255).astype(np.uint8)

        if i >= F_IN0:
            alpha = ease((i - F_IN0) / (F_IN1 - F_IN0))
            s = 1.0 + (SCALE - 1.0) * ease((i - F_SC0) / (F_SC1 - F_SC0))
            img = render_arrow(img, Q_CENTER, s, alpha)
        frames.append(img)

    # final frame: crisp, exact result
    frames[-1] = render_arrow(no_q.copy(), Q_CENTER, SCALE, 1.0)

    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", "-preset", "medium", OUT]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f).tobytes())
    p.stdin.close()
    p.wait()
    if p.returncode != 0:
        raise SystemExit("ffmpeg failed")
    Image.fromarray(frames[-1]).save(f"{ROOT}/output/last_frame.png")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
