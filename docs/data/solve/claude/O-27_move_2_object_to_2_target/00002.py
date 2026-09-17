#!/usr/bin/env python3
"""Move each filled object to its matching dashed outline along a straight line.

Both objects move simultaneously and arrive together. Everything else in the
frame (background, outlines) is left untouched.
"""
import os
import subprocess

import cv2
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
N_FRAMES = 35
FPS = 16


def hue_of(rgb):
    px = np.uint8([[rgb]])
    return int(cv2.cvtColor(px, cv2.COLOR_RGB2HSV)[0, 0, 0])


def hue_dist(a, b):
    d = abs(a - b)
    return min(d, 180 - d)


def main():
    img = np.array(Image.open(FIRST).convert("RGB"))
    h, w = img.shape[:2]

    # Background = most common colour.
    cols, counts = np.unique(img.reshape(-1, 3), axis=0, return_counts=True)
    bg = cols[np.argmax(counts)].astype(np.int32)

    fg = (np.abs(img.astype(np.int32) - bg).sum(axis=2) > 30).astype(np.uint8)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(fg, connectivity=8)

    comps = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        mean_col = img[lab == i].mean(axis=0)
        comps.append(dict(id=i, area=area, hue=hue_of(mean_col.astype(np.uint8))))

    # Objects are the big solid blobs; outline dashes are the small pieces.
    max_area = max(c["area"] for c in comps)
    objects = [c for c in comps if c["area"] > 0.25 * max_area]
    dashes = [c for c in comps if c["area"] <= 0.25 * max_area]

    # Assign every dash to the object with the closest hue.
    for d in dashes:
        d["obj"] = min(objects, key=lambda o: hue_dist(o["hue"], d["hue"]))["id"]

    moves = []
    for o in objects:
        obj_mask = (lab == o["id"]).astype(np.uint8)
        m = cv2.moments(obj_mask, binaryImage=True)
        src = np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]])

        pts = []
        for d in dashes:
            if d["obj"] == o["id"]:
                ys, xs = np.nonzero(lab == d["id"])
                pts.append(np.stack([xs, ys], axis=1))
        pts = np.concatenate(pts).astype(np.int32)
        hull = cv2.convexHull(pts)
        hm = cv2.moments(hull)
        dst = np.array([hm["m10"] / hm["m00"], hm["m01"] / hm["m00"]])

        # Soft-edged cutout of the object (dilate a little to catch AA edge).
        soft = cv2.dilate(obj_mask, np.ones((3, 3), np.uint8))
        alpha = np.zeros((h, w), np.float32)
        # Alpha from colour distance to the background inside the soft mask.
        dist = np.abs(img.astype(np.float32) - bg).sum(axis=2)
        full = np.abs(img[lab == o["id"]].astype(np.float32).mean(axis=0) - bg).sum()
        alpha[soft > 0] = np.clip(dist[soft > 0] / max(full, 1.0), 0, 1)
        alpha[obj_mask > 0] = 1.0
        # Un-blend edge colours from the background to get the pure object colour.
        a3 = alpha[..., None]
        col = np.where(a3 > 0.02,
                       (img.astype(np.float32) - (1 - a3) * bg) / np.maximum(a3, 0.02),
                       0).clip(0, 255)
        moves.append(dict(src=src, dst=dst, alpha=alpha, col=col.astype(np.float32),
                          soft=soft))

    # Static plate: original frame with objects erased.
    plate = img.astype(np.float32).copy()
    for mv in moves:
        plate[mv["soft"] > 0] = bg

    os.makedirs(OUT_DIR, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow", OUT]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    for f in range(N_FRAMES):
        t = f / (N_FRAMES - 1)
        if f == 0:
            frame = img.copy()
        else:
            frame = plate.copy()
            for mv in moves:
                off = (mv["dst"] - mv["src"]) * t
                M = np.float32([[1, 0, off[0]], [0, 1, off[1]]])
                pre = mv["col"] * mv["alpha"][..., None]
                pre_w = cv2.warpAffine(pre, M, (w, h), flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                a_w = cv2.warpAffine(mv["alpha"], M, (w, h), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                frame = frame * (1 - a_w[..., None]) + pre_w
            frame = np.clip(np.round(frame), 0, 255).astype(np.uint8)
        proc.stdin.write(frame.tobytes())

    proc.stdin.close()
    proc.wait()
    print("wrote", OUT)


if __name__ == "__main__":
    main()
