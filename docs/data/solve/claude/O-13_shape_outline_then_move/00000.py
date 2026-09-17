"""Generate the analogy video: bottom-row star becomes outline-only, then moves down."""
import math
import subprocess
import numpy as np
from PIL import Image, ImageDraw

W = H = 1024
FPS = 16
N = 64
GREEN = (30, 153, 71)
STROKE = 5          # outline stroke width (matches the top-row outline arrows)
DROP = 60           # downward move (matches the top-row arrow: y 258 -> 318)

# Star geometry measured from first_frame.png (regular pentagram)
CX, CY = 188.0, 682.0
R_OUT, R_IN = 80.0, 80.0 * 0.382

def star_points(cx, cy):
    pts = []
    for i in range(10):
        ang = -math.pi / 2 + i * math.pi / 5
        r = R_OUT if i % 2 == 0 else R_IN
        pts.append((cx + r * math.cos(ang), cy + r * math.sin(ang)))
    return pts

def ease(t):
    return 0.5 - 0.5 * math.cos(math.pi * min(max(t, 0.0), 1.0))

def draw_outline(img, cx, cy):
    d = ImageDraw.Draw(img)
    pts = star_points(cx, cy)
    d.line(pts + [pts[0]], fill=GREEN, width=STROKE, joint="curve")

def main():
    base = Image.open("/app/first_frame.png").convert("RGB")
    base_np = np.array(base)
    # Mask of the original filled star (bottom-left region)
    star_mask = np.zeros((H, W), bool)
    reg = (base_np[550:800, :330] == GREEN).all(2)
    star_mask[550:800, :330] = reg
    bg_np = base_np.copy()
    bg_np[star_mask] = 255  # scene without the star

    # Timeline (frames): hold, fade fill -> outline, hold, move down, hold
    T0, T1, T2, T3 = 6, 30, 36, 58

    frames = []
    for f in range(N):
        if f <= T0:
            frames.append(base_np.copy()); continue
        if f <= T1:
            a = ease((f - T0) / (T1 - T0))   # 0 -> 1 : fill fades out
            dy = 0.0
        elif f <= T2:
            a, dy = 1.0, 0.0
        elif f <= T3:
            a, dy = 1.0, DROP * ease((f - T2) / (T3 - T2))
        else:
            a, dy = 1.0, float(DROP)
        img = Image.fromarray(bg_np.copy())
        if a < 1.0:
            # partially faded fill
            arr = np.array(img).astype(np.float32)
            fill = np.array(GREEN, np.float32) * (1 - a) + 255.0 * a
            arr[star_mask] = fill
            img = Image.fromarray(arr.round().astype(np.uint8))
        draw_outline(img, CX, CY + dy)
        frames.append(np.array(img))

    frames[0] = base_np.copy()
    Image.fromarray(frames[-1]).save("/app/output/last_frame.png")

    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
           "/app/output/video.mp4"]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fr in frames:
        p.stdin.write(np.ascontiguousarray(fr).tobytes())
    p.stdin.close(); p.wait()
    assert p.returncode == 0

if __name__ == "__main__":
    main()
