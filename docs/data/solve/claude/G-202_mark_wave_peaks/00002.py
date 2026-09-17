"""Detect wave peaks in first_frame.png and animate circling them left to right."""
import subprocess, os
import numpy as np
from PIL import Image, ImageDraw

BASE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(BASE, "first_frame.png")
OUT_DIR = os.path.join(BASE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
N_FRAMES, FPS = 10, 16
RED = (255, 0, 0)
R_CIRCLE, R_DOT, W_CIRCLE = 22, 5, 3


def find_peaks(img):
    """Return (x, y) of local maxima of the curve (minimum image-y per column)."""
    arr = np.array(img.convert("L"))
    ink = arr < 128
    cols = np.where(ink.any(axis=0))[0]
    x0, x1 = cols.min(), cols.max()
    # stroke centre per column: mean of ink rows
    ys = np.array([np.nonzero(ink[:, x])[0].mean() for x in range(x0, x1 + 1)])
    # smooth a little to defeat pixel quantisation, then locate maxima (min y)
    k = np.ones(5) / 5
    ys_s = np.convolve(np.pad(ys, 2, mode="edge"), k, mode="valid")
    peaks = []
    n = len(ys_s)
    i = 1
    while i < n - 1:
        if ys_s[i] < ys_s[i - 1] and ys_s[i] <= ys_s[i + 1]:
            j = i
            while j + 1 < n - 1 and ys_s[j + 1] == ys_s[i]:
                j += 1
            if ys_s[j] < ys_s[j + 1]:          # a true maximum (plateau allowed)
                c = (i + j) // 2
                peaks.append((x0 + c, float(ys[c])))
            i = j + 1
        else:
            i += 1
    return peaks


def draw_step(base, peaks, frame):
    """frame 0: untouched. Each peak gets 3 frames: dot, half arc, full circle."""
    img = base.copy()
    d = ImageDraw.Draw(img)
    for k, (x, y) in enumerate(peaks):
        stage = frame - 3 * k          # <=0 none, 1 dot, 2 dot+arc, >=3 complete
        if stage <= 0:
            continue
        d.ellipse([x - R_DOT, y - R_DOT, x + R_DOT, y + R_DOT], fill=RED)
        box = [x - R_CIRCLE, y - R_CIRCLE, x + R_CIRCLE, y + R_CIRCLE]
        if stage == 2:
            d.arc(box, start=-90, end=90, fill=RED, width=W_CIRCLE)
        elif stage >= 3:
            d.ellipse(box, outline=RED, width=W_CIRCLE)
    return img


def main():
    base = Image.open(SRC).convert("RGB")
    peaks = find_peaks(base)
    print("peaks:", [(x, round(y, 1)) for x, y in peaks])
    os.makedirs(OUT_DIR, exist_ok=True)
    frames_dir = os.path.join(OUT_DIR, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    for f in range(N_FRAMES):
        draw_step(base, peaks, f).save(os.path.join(frames_dir, f"{f:03d}.png"))
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
        "-i", os.path.join(frames_dir, "%03d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", OUT], check=True)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
