"""Generate the analogy video: minus recolors (green->purple), then moves down."""
import numpy as np
from PIL import Image
import subprocess, os, shutil

ROOT = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(ROOT, "first_frame.png")
OUT = os.path.join(ROOT, "output", "video.mp4")
FPS, N = 16, 60

base = np.array(Image.open(FIRST).convert("RGB"))
H, W, _ = base.shape

GREEN = np.array([70, 153, 53], dtype=float)
PURPLE = np.array([91, 30, 153], dtype=float)

# --- Scene geometry (measured from first_frame.png) ---
# minus (D) sprite bbox, incl. outline
MX0, MX1, MY0, MY1 = 115, 276, 662, 703
sprite = base[MY0:MY1, MX0:MX1].copy()
sprite_mask = np.abs(sprite.astype(int) - 255).sum(2) > 30
fill_mask = np.all(np.abs(sprite.astype(int) - GREEN) < 40, axis=2)
# slots (x offsets of the three columns) and vertical move demonstrated on top row
SLOT_DX = [0, 402 - 115, 689 - 115]
MOVE_DY = 361 - 261  # top-row C is 100 px lower than B
# question-mark regions (with margin) to erase
QMARKS = [(463, 502, 654, 709), (750, 789, 654, 709)]


def ease(t):
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(t, 0, 1))


def paste(frame, x, y, color_t):
    """Paste the minus sprite at (x, y) with fill color lerped green->purple by color_t."""
    spr = sprite.copy().astype(float)
    col = GREEN + (PURPLE - GREEN) * color_t
    spr[fill_mask] = col
    h, w = sprite_mask.shape
    region = frame[y:y + h, x:x + w]
    region[sprite_mask] = np.round(spr[sprite_mask]).astype(np.uint8)


def clear(frame, box):
    x0, x1, y0, y1 = box
    frame[y0:y1, x0:x1] = 255


frames = []
half = N // 2
for i in range(N):
    f = base.copy()
    if i == 0:
        frames.append(f)
        continue
    if i < half:
        # Step 1: shape appears in slot 2 and recolors to the demonstrated color.
        t = ease((i - 1) / (half - 2))
        clear(f, QMARKS[0])
        paste(f, MX0 + SLOT_DX[1], MY0, t)
    else:
        # Step 2: recolored shape in slot 3 moves down by the demonstrated amount.
        t = ease((i - half) / (N - 1 - half))
        clear(f, QMARKS[0]); clear(f, QMARKS[1])
        paste(f, MX0 + SLOT_DX[1], MY0, 1.0)
        paste(f, MX0 + SLOT_DX[2], MY0 + int(round(MOVE_DY * t)), 1.0)
    frames.append(f)

tmp = os.path.join(ROOT, "output", "_frames")
shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
for i, f in enumerate(frames):
    Image.fromarray(f).save(os.path.join(tmp, f"{i:04d}.png"))
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                "-i", os.path.join(tmp, "%04d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-crf", "12", "-preset", "slow", OUT], check=True)
shutil.rmtree(tmp)
print("wrote", OUT)
