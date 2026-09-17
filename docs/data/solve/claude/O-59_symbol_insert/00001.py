#!/usr/bin/env python3
"""Insert a blue solid diamond at slot 4: heart slides right to slot 5, then
the diamond fades in above the gap and slides down into slot 4."""
import subprocess
import numpy as np
from PIL import Image

SRC = "/app/first_frame.png"
OUT = "/app/output/video.mp4"
FPS, N = 16, 32

base = np.array(Image.open(SRC).convert("RGB")).astype(np.float32)
H, W, _ = base.shape

# Slot geometry (measured from first frame): boxes 97px wide, pitch 105px.
SLOT_X0 = [254, 359, 464, 569, 674]
SLOT_Y0 = 464
BOX = 97
PITCH = 105
TARGET = 3  # index of position 4

# --- Extract the heart sprite from slot 4 interior (mask of non-white pixels)
def interior(i):
    x0 = SLOT_X0[i] + 2
    y0 = SLOT_Y0 + 2
    return x0, y0, base[y0:y0 + BOX - 4, x0:x0 + BOX - 4]

hx, hy, heart = interior(TARGET)
heart_mask = (np.abs(heart - 255).sum(2) > 0).astype(np.float32)[..., None]

# --- Diamond sprite copied from the reference panel (exact same shape/colour)
panel = base[18:138, 887:1007]
pm = (np.abs(panel - np.array([0, 0, 255], np.float32)).sum(2) < 60)
ys, xs = np.where(pm)
dia = panel[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
dia_mask = pm[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.float32)[..., None]
dh, dw = dia_mask.shape[:2]
# Final diamond position: centred in slot 4
cx = SLOT_X0[TARGET] + BOX / 2.0
cy = SLOT_Y0 + BOX / 2.0
dia_x_final = int(round(cx - dw / 2.0))
dia_y_final = int(round(cy - dh / 2.0))

# Background with the heart removed from slot 4 (interior becomes white)
clean = base.copy()
clean[hy:hy + heart.shape[0], hx:hx + heart.shape[1]] = (
    heart * (1 - heart_mask) + 255 * heart_mask)

def blit(frame, sprite, mask, x, y, alpha=1.0):
    h, w = mask.shape[:2]
    x, y = int(round(x)), int(round(y))
    region = frame[y:y + h, x:x + w]
    a = mask * alpha
    frame[y:y + h, x:x + w] = region * (1 - a) + sprite * a

def ease(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)  # smoothstep

SLIDE_END = 14      # frames 0..14: heart slides right
DROP_START = 15     # frames 15..31: diamond fades in and drops
RISE = 90           # px above the final position where the diamond starts

frames = []
for f in range(N):
    if f == 0:
        frames.append(base.astype(np.uint8))
        continue
    frame = clean.copy()
    # Heart position
    t1 = ease(min(f, SLIDE_END) / SLIDE_END)
    blit(frame, heart, heart_mask, hx + PITCH * t1, hy)
    # Diamond
    if f >= DROP_START:
        t2 = ease((f - DROP_START) / (N - 1 - DROP_START))
        alpha = min(1.0, t2 * 2.0)            # fade in during the first half
        y = dia_y_final - RISE * (1 - t2)
        blit(frame, dia, dia_mask, dia_x_final, y, alpha)
    frames.append(np.clip(frame, 0, 255).astype(np.uint8))

# Encode with ffmpeg (H.264, yuv420p)
cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
       "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
       "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
       "-movflags", "+faststart", OUT]
p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
for fr in frames:
    p.stdin.write(fr.tobytes())
p.stdin.close()
p.wait()
print("wrote", OUT, len(frames), "frames")
