#!/usr/bin/env python3
"""Generate the object-counting video from first_frame.png.

Pipeline: detect every filled+outlined shape as a connected component of
non-background pixels, count them in reading order (top-to-bottom,
left-to-right) while highlighting each one's border, then show "Count: N"
in the centre of the image.  All pixels not touched by the highlight or the
text stay identical to first_frame.png.
"""
import os
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

APP = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(APP, "first_frame.png")
OUT_DIR = os.path.join(APP, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")

FPS = 16
N_FRAMES = 35
HIGHLIGHT = (255, 40, 40)      # RGB colour used to highlight a border
HIGHLIGHT_THICK = 6
TEXT_COLOR = (0, 0, 0)


def detect_objects(img):
    """Return a list of outer contours, one per object, in reading order."""
    h, w, _ = img.shape
    # Background is the dominant colour (sampled from the corners).
    corners = np.array([img[0, 0], img[0, -1], img[-1, 0], img[-1, -1]])
    bg = np.median(corners, axis=0).astype(np.int32)
    diff = np.abs(img.astype(np.int32) - bg).sum(axis=2)
    mask = (diff > 30).astype(np.uint8) * 255
    # Close tiny gaps in outlines so each shape is one component.
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    objs = []
    for c in contours:
        if cv2.contourArea(c) < 80:
            continue
        m = cv2.moments(c)
        cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
        objs.append((c, cx, cy))
    # Reading order: group into horizontal bands, then left to right.
    band = 120
    objs.sort(key=lambda o: (int(o[2] // band), o[1]))
    return [o[0] for o in objs]


def load_font(size):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def draw_highlight(frame, contour, progress):
    """Draw the border highlight; progress in (0,1] grows the stroke."""
    thick = max(2, int(round(HIGHLIGHT_THICK * progress)))
    cv2.drawContours(frame, [contour], -1, HIGHLIGHT, thick, lineType=cv2.LINE_AA)


def draw_text(frame, text):
    pil = Image.fromarray(frame)
    d = ImageDraw.Draw(pil)
    font = load_font(72)
    w, h = pil.size
    bbox = d.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (w - tw) // 2 - bbox[0]
    y = (h - th) // 2 - bbox[1]
    d.text((x, y), text, font=font, fill=TEXT_COLOR)
    return np.array(pil)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    base = np.array(Image.open(FIRST).convert("RGB"))
    h, w, _ = base.shape
    contours = detect_objects(base)
    n = len(contours)
    print(f"Detected {n} objects")

    # Timeline: frame 0 untouched; counting spans frames 1..count_end;
    # the remaining frames show the final "Count: N".
    text_frames = max(8, N_FRAMES // 4)
    count_start, count_end = 1, N_FRAMES - text_frames  # exclusive end
    per_obj = (count_end - count_start) / max(n, 1)

    frames = []
    for f in range(N_FRAMES):
        frame = base.copy()
        if f >= count_start:
            for i, c in enumerate(contours):
                start = count_start + i * per_obj
                if f + 1e-9 < start:
                    break
                # grow the stroke over the first few frames of its slot
                progress = min(1.0, (f - start + 1) / max(1.0, per_obj * 0.5))
                draw_highlight(frame, c, progress)
        if f >= count_end:
            frame = draw_text(frame, f"Count: {n}")
        frames.append(frame)

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(FPS),
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
        "-r", str(FPS), OUT,
    ]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fr in frames:
        p.stdin.write(np.ascontiguousarray(fr).tobytes())
    p.stdin.close()
    p.wait()
    if p.returncode != 0:
        raise SystemExit("ffmpeg failed")
    print(f"Wrote {OUT} ({len(frames)} frames @ {FPS} fps)")


if __name__ == "__main__":
    main()
