#!/usr/bin/env python3
"""Regenerate /app/output/video.mp4: lights 2-4 turn on, light 6 turns off."""
import subprocess, numpy as np
from PIL import Image

SRC = "/app/first_frame.png"
OUT = "/app/output/video.mp4"
W = H = 1024
FPS = 16
N_FRAMES = 35

WHITE = (255, 255, 255)
OFF_FILL, OFF_RING = (128, 128, 128), (64, 64, 64)
ON_FILL, ON_RING, GLOW = (255, 255, 0), (255, 200, 0), (255, 255, 179)
R_GLOW, R_RING, RING_W = 42, 34, 3
CY = 512

base = np.array(Image.open(SRC).convert("RGB"))


def detect_lights(img):
    """Return list of (cx, is_on) for the 6 lights, left to right."""
    mask = (img.astype(int).sum(2) < 750).any(0)
    cols = np.where(mask)[0]
    segs, s, p = [], cols[0], cols[0]
    for c in cols[1:]:
        if c != p + 1:
            segs.append((s, p)); s = c
        p = c
    segs.append((s, p))
    lights = []
    for a, b in segs:
        cx = (a + b) // 2
        on = img[CY, cx, 2] < 100 and img[CY, cx, 0] > 200
        lights.append((int(cx), bool(on)))
    return lights


def lerp(c0, c1, t):
    return tuple(int(round(a + (b - a) * t)) for a, b in zip(c0, c1))


def ease(t):
    return t * t * (3 - 2 * t)


def build_masks(img, lights):
    """Derive glow/ring/fill masks (in patch coords) from an on light and an off light."""
    on_cx = next(cx for cx, on in lights if on)
    off_cx = next(cx for cx, on in lights if not on)
    def patch(cx):
        return img[CY - R_GLOW:CY + R_GLOW + 1, cx - R_GLOW:cx + R_GLOW + 1]
    on, off = patch(on_cx), patch(off_cx)
    glow = (on == GLOW).all(2)
    ring = (on == ON_RING).all(2)
    fill = (on == ON_FILL).all(2)
    assert np.array_equal(ring, (off == OFF_RING).all(2))
    assert np.array_equal(fill, (off == OFF_FILL).all(2))
    return glow, ring, fill


def render_light(masks, t):
    """Return an RGB patch for state t (0=off, 1=on); exact at endpoints."""
    glow, ring, fill = masks
    p = np.full(glow.shape + (3,), 255, np.uint8)
    if t > 0:
        p[glow] = lerp(WHITE, GLOW, t)
    p[ring] = lerp(OFF_RING, ON_RING, t)
    p[fill] = lerp(OFF_FILL, ON_FILL, t)
    return p


def state_at(frame, start, end, s0, s1):
    if frame <= start:
        return s0
    if frame >= end:
        return s1
    return s0 + (s1 - s0) * ease((frame - start) / (end - start))


def main():
    lights = detect_lights(base)
    assert len(lights) == 6, lights
    masks = build_masks(base, lights)
    target = [False, True, True, True, False, False]
    # Schedule: one light changes at a time, left to right, spread over the clip.
    changes = [i for i in range(6) if lights[i][1] != target[i]]
    seg = (N_FRAMES - 3) / max(len(changes), 1)
    schedule = {i: (1 + k * seg, 1 + (k + 1) * seg) for k, i in enumerate(changes)}

    frames = []
    for f in range(N_FRAMES):
        img = base.copy()
        for i, (cx, on) in enumerate(lights):
            if i not in schedule:
                continue
            a, b = schedule[i]
            t = state_at(f, a, b, float(on), float(target[i]))
            img[CY - R_GLOW:CY + R_GLOW + 1, cx - R_GLOW:cx + R_GLOW + 1] = render_light(masks, t)
        frames.append(img)

    assert np.array_equal(frames[0], base), "first frame must be unchanged"

    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "4", "-preset", "slow", "-tune", "stillimage", OUT]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fr in frames:
        p.stdin.write(fr.tobytes())
    p.stdin.close()
    p.wait()
    assert p.returncode == 0
    print("wrote", OUT, len(frames), "frames")


if __name__ == "__main__":
    main()
