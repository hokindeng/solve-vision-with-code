"""Generate video: highlight the correct candidate (large yellow square, card 3)
with a red ring that is drawn progressively around it."""
import numpy as np
from PIL import Image, ImageDraw
import subprocess, os

BASE = "/app/first_frame.png"
OUT = "/app/output/video.mp4"
N_FRAMES, FPS, SS = 60, 16, 4          # SS = supersample factor for anti-aliasing

# Bottom row, third card (large yellow square) — the next item after small-large-small is large.
CX, CY = 626, 854
RADIUS, THICK = 100, 7
RED = (230, 30, 30)

base = np.array(Image.open(BASE).convert("RGB")).astype(np.float32)
H, W = base.shape[:2]

def ring_mask(sweep_deg):
    """Anti-aliased mask (0..1) of an arc from -90deg sweeping clockwise by sweep_deg."""
    if sweep_deg <= 0:
        return np.zeros((H, W), np.float32)
    big = Image.new("L", (W * SS, H * SS), 0)
    d = ImageDraw.Draw(big)
    r_out = RADIUS + THICK / 2
    bbox = [(CX - r_out) * SS, (CY - r_out) * SS, (CX + r_out) * SS, (CY + r_out) * SS]
    start = -90
    if sweep_deg >= 360:
        d.ellipse(bbox, outline=255, width=THICK * SS)
    else:
        d.arc(bbox, start, start + sweep_deg, fill=255, width=THICK * SS)
        # round caps
        for ang in (start, start + sweep_deg):
            a = np.deg2rad(ang)
            px, py = (CX + RADIUS * np.cos(a)) * SS, (CY + RADIUS * np.sin(a)) * SS
            rr = THICK * SS / 2
            d.ellipse([px - rr, py - rr, px + rr, py + rr], fill=255)
    m = big.resize((W, H), Image.LANCZOS)
    return np.asarray(m).astype(np.float32) / 255.0

def ease(t):
    return 0.5 - 0.5 * np.cos(np.pi * t)

frames = []
START, END = 6, 50   # ring drawn between these frames, then held
for i in range(N_FRAMES):
    if i < START:
        sweep = 0.0
    elif i >= END:
        sweep = 360.0
    else:
        sweep = 360.0 * ease((i - START) / (END - START))
    m = ring_mask(sweep)[..., None]
    img = base * (1 - m) + np.array(RED, np.float32) * m
    frames.append(np.clip(img, 0, 255).astype(np.uint8))

# Frame 0 must be exactly the first frame.
frames[0] = base.astype(np.uint8)

tmp = "/app/output/_frames"
os.makedirs(tmp, exist_ok=True)
for i, f in enumerate(frames):
    Image.fromarray(f).save(f"{tmp}/{i:04d}.png")
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                "-i", f"{tmp}/%04d.png", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-crf", "16", "-r", str(FPS), OUT], check=True)
for fn in os.listdir(tmp):
    os.remove(os.path.join(tmp, fn))
os.rmdir(tmp)
print("wrote", OUT)
