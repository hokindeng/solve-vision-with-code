import os, subprocess, tempfile
import numpy as np
from PIL import Image, ImageDraw

SRC = "/app/first_frame.png"
OUT = "/app/output/video.mp4"
N_FRAMES, FPS = 10, 16
RED = (255, 0, 0)
R_CIRCLE, R_DOT, W = 22, 5, 3


def find_peaks(img):
    """Per-column wave height from the black stroke; peaks = local maxima
    (smaller y), with flat plateaus collapsed to their center column."""
    g = np.array(img.convert("L")) < 128
    xs = np.nonzero(g.any(0))[0]
    y = np.array([np.nonzero(g[:, x])[0].mean() for x in xs])  # center of stroke
    peaks = []
    i, n = 0, len(xs)
    while i < n:
        j = i
        while j + 1 < n and y[j + 1] == y[i]:
            j += 1
        left_ok = i > 0 and y[i - 1] > y[i]
        right_ok = j + 1 < n and y[j + 1] > y[i]
        if left_ok and right_ok:
            c = (i + j) // 2
            # refine: center of topmost pixel run inside the plateau
            top = min(np.nonzero(g[:, xs[k]])[0].min() for k in range(i, j + 1))
            ks = [k for k in range(i, j + 1) if np.nonzero(g[:, xs[k]])[0].min() == top]
            c = ks[len(ks) // 2]
            peaks.append((int(xs[c]), int(round(y[c]))))
        i = j + 1
    return peaks


def draw_step(base, peaks, k, frac):
    """First k peaks fully marked; peak k drawn partially (frac in (0,1])."""
    im = base.copy()
    d = ImageDraw.Draw(im)
    for idx, (px, py) in enumerate(peaks):
        if idx < k:
            d.ellipse([px - R_CIRCLE, py - R_CIRCLE, px + R_CIRCLE, py + R_CIRCLE], outline=RED, width=W)
            d.ellipse([px - R_DOT, py - R_DOT, px + R_DOT, py + R_DOT], fill=RED)
        elif idx == k and frac > 0:
            d.ellipse([px - R_DOT, py - R_DOT, px + R_DOT, py + R_DOT], fill=RED)
            if frac >= 1:
                d.ellipse([px - R_CIRCLE, py - R_CIRCLE, px + R_CIRCLE, py + R_CIRCLE], outline=RED, width=W)
            else:
                d.arc([px - R_CIRCLE, py - R_CIRCLE, px + R_CIRCLE, py + R_CIRCLE],
                      start=-90, end=-90 + 360 * frac, fill=RED, width=W)
    return im


def main():
    base = Image.open(SRC).convert("RGB")
    peaks = sorted(find_peaks(base))
    print("peaks:", peaks)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    steps = N_FRAMES - 1  # frames 1..9 animate; frame 0 is the original
    per = steps / len(peaks)
    frames = [base.copy()]
    for f in range(1, N_FRAMES):
        t = f / per  # progress in units of peaks
        k = min(int(np.floor(t - 1e-9)), len(peaks) - 1)
        frac = t - k
        if f == N_FRAMES - 1:
            k, frac = len(peaks), 0  # final: everything complete
        frames.append(draw_step(base, peaks, k, min(frac, 1.0)))
    with tempfile.TemporaryDirectory() as td:
        for i, fr in enumerate(frames):
            fr.save(f"{td}/{i:03d}.png")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                        "-i", f"{td}/%03d.png", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-qp", "0", "-preset", "slow", OUT], check=True)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
