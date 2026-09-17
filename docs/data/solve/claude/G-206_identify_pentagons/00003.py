import os
import subprocess
import numpy as np
import cv2
from PIL import Image, ImageDraw

SRC = "/app/first_frame.png"
OUT_DIR = "/app/output"
OUT = os.path.join(OUT_DIR, "video.mp4")
FPS = 16
N_FRAMES = 30
SS = 4  # supersampling for anti-aliased circle


def find_pentagon(img):
    """Return (cx, cy, radius_to_farthest_vertex) of the polygon with 5 sides."""
    bg = img[0, 0].astype(int)
    diff = np.abs(img.astype(int) - bg).sum(axis=2)
    mask = (diff > 30).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 500:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        candidates.append((len(approx), c, approx))
    pents = [t for t in candidates if t[0] == 5]
    if len(pents) != 1:
        # fall back: pick the shape whose vertex count is closest to 5
        pents = sorted(candidates, key=lambda t: abs(t[0] - 5))[:1]
    _, c, approx = pents[0]
    pts = c.reshape(-1, 2).astype(float)
    (cx, cy), r = cv2.minEnclosingCircle(c)
    return cx, cy, r


def ease(t):
    return 1 - (1 - t) ** 3  # ease-out cubic


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    base = np.array(Image.open(SRC).convert("RGB"))
    h, w = base.shape[:2]
    cx, cy, r = find_pentagon(base)
    final_r = r + 18
    width = 6

    frames = []
    for i in range(N_FRAMES):
        frame = base.copy()
        if i > 0:
            t = i / (N_FRAMES - 1)
            rad = final_r * ease(t)
            # draw circle on transparent supersampled layer, then composite
            layer = Image.new("RGBA", (w * SS, h * SS), (0, 0, 0, 0))
            d = ImageDraw.Draw(layer)
            bbox = [
                (cx - rad) * SS, (cy - rad) * SS,
                (cx + rad) * SS, (cy + rad) * SS,
            ]
            d.ellipse(bbox, outline=(220, 20, 20, 255), width=width * SS)
            layer = layer.resize((w, h), Image.LANCZOS)
            lay = np.array(layer).astype(float)
            a = lay[..., 3:4] / 255.0
            frame = (frame * (1 - a) + lay[..., :3] * a).round().astype(np.uint8)
        frames.append(frame)

    tmp_dir = os.path.join(OUT_DIR, "_frames")
    os.makedirs(tmp_dir, exist_ok=True)
    for i, f in enumerate(frames):
        Image.fromarray(f).save(os.path.join(tmp_dir, f"{i:04d}.png"))
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
        "-i", os.path.join(tmp_dir, "%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12",
        "-r", str(FPS), OUT,
    ], check=True)
    for fn in os.listdir(tmp_dir):
        os.remove(os.path.join(tmp_dir, fn))
    os.rmdir(tmp_dir)
    print(f"pentagon at ({cx:.1f}, {cy:.1f}) r={r:.1f}; wrote {OUT}")


if __name__ == "__main__":
    main()
