"""Generate video: two balls move toward each other and merge with additive color mixing."""
import os
import subprocess
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
FPS, N_FRAMES = 16, 80

WHITE = np.array([255, 255, 255], np.uint8)
BLACK = np.array([0, 0, 0], np.uint8)


def detect_balls(img):
    """Find the filled colored discs (non-white, non-black uniform regions)."""
    flat = img.reshape(-1, 3)
    cols, counts = np.unique(flat, axis=0, return_counts=True)
    balls = []
    for c, n in zip(cols, counts):
        if n < 500 or np.all(c == 255) or np.all(c == 0):
            continue
        m = np.all(img == c, axis=2)
        ys, xs = np.nonzero(m)
        cx = (xs.min() + xs.max()) / 2.0
        cy = (ys.min() + ys.max()) / 2.0
        r_fill = (xs.max() - xs.min() + 1) / 2.0
        balls.append(dict(color=c.astype(np.float32), cx=cx, cy=cy, r=r_fill))
    assert len(balls) == 2, balls
    return balls


def measure_outline(img, b):
    """Outline thickness: count black pixels just left of the fill on the center row."""
    y = int(round(b["cy"]))
    x = int(round(b["cx"] - b["r"])) - 1
    t = 0
    while x >= 0 and np.all(img[y, x] == 0):
        t += 1
        x -= 1
    return t


def render(base, balls, r_fill, r_outer, H, W):
    """Draw balls additively over the background (background ball pixels removed)."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    frame = base.astype(np.float32).copy()
    light = np.zeros_like(frame)
    fill_any = np.zeros((H, W), bool)
    outline_any = np.zeros((H, W), bool)
    for b in balls:
        d = np.hypot(xx - b["cx"], yy - b["cy"])
        fill = d < r_fill
        ring = (d >= r_fill) & (d < r_outer)
        light[fill] += b["color"]
        fill_any |= fill
        outline_any |= ring
    # Additive light mixing: sum of the two colours, clipped to displayable range.
    frame[fill_any] = np.clip(light[fill_any], 0, 255)
    # Black outline drawn on top; where a ring crosses the other's fill, keep it visible.
    frame[outline_any] = 0
    return frame.astype(np.uint8)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    img = np.array(Image.open(FIRST).convert("RGB"))
    H, W = img.shape[:2]
    balls = detect_balls(img)
    r_fill = balls[0]["r"]
    t_outline = measure_outline(img, balls[0])
    r_outer = r_fill + t_outline

    # Background with both balls (fill + outline) erased to white.
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    base = img.copy()
    for b in balls:
        d = np.hypot(xx - b["cx"], yy - b["cy"])
        base[d < r_outer + 1.0] = WHITE

    # Sanity: re-rendered frame 0 must match the original exactly.
    f0 = render(base, balls, r_fill, r_outer, H, W)
    diff = int((f0 != img).any(axis=2).sum())
    print(f"frame0 mismatch pixels: {diff}")

    mx = (balls[0]["cx"] + balls[1]["cx"]) / 2.0
    my = (balls[0]["cy"] + balls[1]["cy"]) / 2.0
    starts = [(b["cx"], b["cy"]) for b in balls]

    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", "-preset", "medium",
         "-movflags", "+faststart", OUT],
        stdin=subprocess.PIPE)
    for i in range(N_FRAMES):
        t = i / (N_FRAMES - 1)  # linear: constant, equal speed for both balls
        cur = []
        for b, (sx, sy) in zip(balls, starts):
            cur.append(dict(color=b["color"], cx=sx + (mx - sx) * t, cy=sy + (my - sy) * t))
        frame = img if i == 0 else render(base, cur, r_fill, r_outer, H, W)
        ff.stdin.write(np.ascontiguousarray(frame).tobytes())
    ff.stdin.close()
    ff.wait()
    print("wrote", OUT)


if __name__ == "__main__":
    main()
