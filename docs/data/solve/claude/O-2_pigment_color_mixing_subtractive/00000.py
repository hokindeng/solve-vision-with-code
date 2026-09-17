import os
import numpy as np
from PIL import Image
import subprocess

FIRST = "/app/first_frame.png"
OUT_DIR = "/app/output"
OUT = os.path.join(OUT_DIR, "video.mp4")
FPS = 16
N_FRAMES = 44

LEFT = np.array([204, 207, 223], dtype=np.float64)
RIGHT = np.array([39, 232, 220], dtype=np.float64)
MIXED = np.round(LEFT * RIGHT / 255.0).astype(np.uint8)  # subtractive (multiply)

def find_mixing_zone(img):
    """Boolean mask of the white interior enclosed by the black square border.

    Flood-fills from the image centre over non-black pixels; the black border
    stops the fill, so the result is exactly the pixels inside the zone.
    """
    import cv2
    blk = np.all(img < 40, axis=2)
    h, w = blk.shape
    # 1-px black border around the image bounds so the fill cannot leak out
    notblk = (~blk).astype(np.uint8)
    ff_mask = np.zeros((h + 2, w + 2), np.uint8)
    seed = (w // 2, h // 2)
    cv2.floodFill(notblk, ff_mask, seed, 2, loDiff=0, upDiff=0,
                  flags=4 | cv2.FLOODFILL_FIXED_RANGE)
    return notblk == 2

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    base = np.array(Image.open(FIRST).convert("RGB"))
    zone = find_mixing_zone(base)
    interior0 = base[zone].astype(np.float64)
    target = MIXED.astype(np.float64)

    frames = []
    for i in range(N_FRAMES):
        t = i / (N_FRAMES - 1)
        s = t * t * (3 - 2 * t)  # smoothstep easing
        f = base.copy()
        f[zone] = np.round(interior0 * (1 - s) + target * s).astype(np.uint8)
        if i == N_FRAMES - 1:
            f[zone] = MIXED
        frames.append(f)

    h, w = base.shape[:2]
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow", OUT]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(f.tobytes())
    p.stdin.close()
    p.wait()
    print("wrote", OUT, "mixed color", MIXED.tolist())

if __name__ == "__main__":
    main()
