"""Delete the symbol at position 4: fade it out, then slide the later symbols left."""
import subprocess, numpy as np
from PIL import Image

SRC = "/app/first_frame.png"
OUT = "/app/output/video.mp4"
W = H = 1024
FPS = 16
N_FRAMES = 45
FADE_FRAMES = 18                 # frames 0..17: fade; 18..44: slide
TARGET = 4                       # 1-based position to delete
PITCH = 105                      # cell spacing in px
CELL_X0 = [44 + PITCH * i for i in range(9)]   # left border column of each cell
BOX_TOP, BOX_H = 464, 97         # gray box rows 464..560
BG = np.array([255, 255, 255], np.float32)

def interior(i):
    """Interior (inside the gray border) of cell i (0-based) as a slice tuple."""
    x0 = CELL_X0[i]
    return slice(BOX_TOP + 1, BOX_TOP + BOX_H - 1), slice(x0 + 1, x0 + BOX_H - 1)

def ease(t):
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(t, 0, 1))

first = np.array(Image.open(SRC).convert("RGB")).astype(np.float32)

# Base frame: boxes + labels + symbols 1..3; interiors of cells 4..9 cleared to background.
base = first.copy()
for i in range(TARGET - 1, 9):
    base[interior(i)] = BG

# Target symbol layer (cell 4) as it sits in place.
tgt = first.copy()
tgt_mask = np.zeros((H, W), bool)
tgt_mask[interior(TARGET - 1)] = True
tgt_mask &= (first != 255).any(2)

# Sliding layer: RGBA of symbols in cells 5..9 (alpha 1 where not background).
slide_rgba = np.zeros((H, W, 4), np.float32)
for i in range(TARGET, 9):
    ys, xs = interior(i)
    patch = first[ys, xs]
    alpha = (patch != 255).any(2).astype(np.float32)
    slide_rgba[ys, xs, :3] = patch
    slide_rgba[ys, xs, 3] = alpha
slide_img = Image.fromarray(np.dstack([slide_rgba[..., :3].astype(np.uint8),
                                       (slide_rgba[..., 3] * 255).astype(np.uint8)]), "RGBA")

def composite(dst, rgba):
    a = rgba[..., 3:4] / 255.0
    return dst * (1 - a) + rgba[..., :3] * a

frames = []
for f in range(N_FRAMES):
    frame = base.copy()
    if f < FADE_FRAMES:
        k = 1.0 - f / (FADE_FRAMES - 1)          # 1 -> 0
        faded = BG + (tgt - BG) * k
        frame[tgt_mask] = faded[tgt_mask]
        dx = 0.0
    else:
        t = (f - FADE_FRAMES) / (N_FRAMES - 1 - FADE_FRAMES)
        dx = -PITCH * float(ease(t))
    shifted = slide_img.transform((W, H), Image.AFFINE, (1, 0, -dx, 0, 1, 0), resample=Image.BILINEAR)
    frame = composite(frame, np.array(shifted).astype(np.float32))
    frames.append(np.clip(frame + 0.5, 0, 255).astype(np.uint8))

# Frame 0 must be exactly the input; slide layer at dx=0 reproduces it, but enforce anyway.
frames[0] = first.astype(np.uint8)

cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
       "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
       "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow", OUT]
p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
for fr in frames:
    p.stdin.write(fr.tobytes())
p.stdin.close(); p.wait()
Image.fromarray(frames[-1]).save("/app/output/last_frame.png")
print("wrote", OUT, "frames:", len(frames))
