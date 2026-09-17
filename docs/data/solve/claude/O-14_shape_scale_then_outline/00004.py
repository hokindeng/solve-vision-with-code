#!/usr/bin/env python3
"""Generate /app/output/video.mp4: bottom-left rectangle scales down (step 1),
then its thick fill-stroke converts to a thin outline (step 2). Everything else
stays identical to first_frame.png."""
import subprocess, numpy as np
from PIL import Image, ImageDraw

SRC = "/app/first_frame.png"
OUT = "/app/output/video.mp4"
N, FPS, S = 16, 16, 4                       # frames, fps, supersample factor
PURPLE, WHITE = (119, 53, 153), (255, 255, 255)

# Rectangle in first frame: pixel-edge bbox [74,273) x [655,710), stroke 6.
X0, X1, Y0, Y1 = 74, 273, 655, 710
CX, CY = (X0 + X1) / 2, (Y0 + Y1) / 2
W0, H0, STROKE0 = X1 - X0, Y1 - Y0, 6
SCALE = 139 / 199                            # from the example row (199 -> 139)
W1, H1, STROKE1 = 139, 39, 4                 # final size / thin outline (example: ~4px)

def smooth(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)

def render(frame_idx, base):
    if frame_idx == 0:
        return base.copy()
    k = frame_idx
    if k <= 8:                                # step 1: scale change
        a, b = smooth(k / 8), 0.0
    else:                                     # step 2: fill -> outline
        a, b = 1.0, smooth((k - 8) / 7)
    w = W0 + (W1 - W0) * a
    h = H0 + (H1 - H0) * a
    st = STROKE0 + (STROKE1 - STROKE0) * b
    if k == N - 1:                            # crisp integers on the final frame
        w, h, st = W1, H1, STROKE1

    img = base.copy()
    # clear original rectangle region (pure white background there)
    region = (X0, Y0, X1, Y1)
    ImageDraw.Draw(img).rectangle((X0, Y0, X1 - 1, Y1 - 1), fill=WHITE)

    # supersampled rendering of the new rectangle within the region
    rw, rh = (X1 - X0) * S, (Y1 - Y0) * S
    hi = Image.new("RGB", (rw, rh), WHITE)
    d = ImageDraw.Draw(hi)
    l = (CX - w / 2 - X0) * S; r = (CX + w / 2 - X0) * S
    t = (CY - h / 2 - Y0) * S; btm = (CY + h / 2 - Y0) * S
    d.rectangle((round(l), round(t), round(r) - 1, round(btm) - 1), fill=PURPLE)
    sS = st * S
    d.rectangle((round(l + sS), round(t + sS), round(r - sS) - 1, round(btm - sS) - 1), fill=WHITE)
    lo = hi.resize((X1 - X0, Y1 - Y0), Image.BOX)
    img.paste(lo, (X0, Y0))
    return img

def main():
    base = Image.open(SRC).convert("RGB")
    frames = [render(i, base) for i in range(N)]
    for i, f in enumerate(frames):
        f.save(f"/app/output/frame_{i:02d}.png")
    cmd = ["ffmpeg", "-y", "-framerate", str(FPS), "-i", "/app/output/frame_%02d.png",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
           "-r", str(FPS), OUT]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("wrote", OUT)

if __name__ == "__main__":
    main()
