#!/usr/bin/env python3
"""Generate the candidate-check / piece-fitting video from first_frame.png."""
import os, subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont

APP = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(APP, "first_frame.png")
OUT = os.path.join(APP, "output", "video.mp4")
FPS, N_FRAMES = 16, 99

BG = (245, 245, 245)
PURPLE = (138, 43, 226)
OUTLINE = (50, 100, 140)
DASH_RED = (255, 100, 100)
BOX_FILL, BOX_EDGE, LABEL = 230, 180, 100
GREEN, GREEN_TINT = (40, 165, 60), (208, 238, 210)
RED, RED_TINT = (220, 50, 50), (245, 210, 210)
HIGHLIGHT = (30, 120, 230)

# Structure grid (pitch 52, cell rect x0..x0+50) and gap cells
S_PITCH, S_SIZE = 52, 50
GAP_ORIGIN = (486, 383)
GAP_SHAPE = {(0, 0), (1, 0), (0, 1)}

# Candidate boxes (pitch 38, cell rect x0..x0+36)
C_PITCH, C_SIZE = 38, 36
BOXES = [(50 + 246 * k, 829, 235 + 246 * k, 1014) for k in range(4)]
CANDS = [
    dict(origin=(85, 883), shape={(0, 0), (1, 0), (1, 1), (2, 1)}),
    dict(origin=(350, 902), shape={(0, 0), (1, 0)}),
    dict(origin=(596, 883), shape={(0, 0), (1, 0), (1, 1)}),
    dict(origin=(842, 883), shape={(0, 0), (1, 0), (0, 1)}),
]
for c in CANDS:
    c["match"] = c["shape"] == GAP_SHAPE

# Timeline
T_HL, T_PREV, T_JUDGE = 4, 8, 6
T_CAND = T_HL + T_PREV + T_JUDGE          # 18 frames per candidate
T_START = 1                                 # frame 0 = original
T_MOVE_START = T_START + 4 * T_CAND          # 73
T_MOVE = 20
T_LAND = T_MOVE_START + T_MOVE               # 93

try:
    FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 34)
except Exception:
    FONT = None


def ease(t):
    return 0.5 - 0.5 * np.cos(np.pi * t)


def cell_rect(origin, pitch, size, cx, cy):
    x0 = origin[0] + pitch * cx
    y0 = origin[1] + pitch * cy
    return (x0, y0, x0 + size, y0 + size)


def draw_cells(draw, origin, pitch, size, shape, fill=PURPLE, outline=OUTLINE):
    for cx, cy in shape:
        draw.rectangle(cell_rect(origin, pitch, size, cx, cy), fill=fill, outline=outline)


def color_box(img, k, tint, edge, remove_piece=False):
    """Recolor candidate box k: gray fill -> tint (label antialias preserved), edge -> colored."""
    x0, y0, x1, y1 = BOXES[k]
    a = np.array(img)
    reg = a[y0:y1 + 1, x0:x1 + 1].astype(np.float32)
    gray = (reg[..., 0] == reg[..., 1]) & (reg[..., 1] == reg[..., 2])
    g = reg[..., 0]
    inside = gray & (g <= BOX_FILL) & (g >= 90) & (np.abs(g - BOX_EDGE) > 0.5)
    # label alpha from anti-aliased gray value
    alpha = np.clip((BOX_FILL - g) / (BOX_FILL - LABEL), 0, 1)[..., None]
    tinted = alpha * np.array(LABEL, np.float32) + (1 - alpha) * np.array(tint, np.float32)
    reg[inside] = tinted[inside]
    if remove_piece:
        notgray = ~gray
        reg[notgray] = np.array(tint, np.float32)
    a[y0:y1 + 1, x0:x1 + 1] = np.clip(reg, 0, 255).astype(np.uint8)
    img = Image.fromarray(a)
    d = ImageDraw.Draw(img)
    d.rectangle((x0, y0, x1, y1), outline=edge, width=2)
    return img


def draw_mark(draw, k, ok):
    x0, y0, x1, y1 = BOXES[k]
    cx, cy = x1 - 26, y0 + 26
    col = GREEN if ok else RED
    if ok:
        draw.line([(cx - 14, cy + 1), (cx - 4, cy + 11), (cx + 15, cy - 12)], fill=col, width=5, joint="curve")
    else:
        draw.line([(cx - 12, cy - 12), (cx + 12, cy + 12)], fill=col, width=5)
        draw.line([(cx - 12, cy + 12), (cx + 12, cy - 12)], fill=col, width=5)


def draw_highlight(draw, k):
    x0, y0, x1, y1 = BOXES[k]
    draw.rectangle((x0 - 5, y0 - 5, x1 + 5, y1 + 5), outline=HIGHLIGHT, width=3)


def preview_overlay(base, shape, alpha):
    over = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    a = int(255 * alpha)
    for cx, cy in shape:
        d.rectangle(cell_rect(GAP_ORIGIN, S_PITCH, S_SIZE, cx, cy),
                    fill=PURPLE + (int(a * 0.55),), outline=OUTLINE + (a,))
    return Image.alpha_composite(base.convert("RGBA"), over).convert("RGB")


def remove_dashes(img):
    a = np.array(img)
    m = (a == np.array(DASH_RED, np.uint8)).all(2)
    a[m] = BG
    return Image.fromarray(a)


def main():
    first = Image.open(FIRST).convert("RGB")
    frames = []
    for f in range(N_FRAMES):
        img = first.copy()
        # judged states so far
        judged = [k for k in range(4) if f >= T_START + k * T_CAND + T_HL + T_PREV]
        moving = f >= T_MOVE_START
        for k in judged:
            ok = CANDS[k]["match"]
            img = color_box(img, k, GREEN_TINT if ok else RED_TINT, GREEN if ok else RED,
                            remove_piece=(ok and moving))
        d = ImageDraw.Draw(img)
        for k in judged:
            draw_mark(d, k, CANDS[k]["match"])

        if T_START <= f < T_MOVE_START:
            k = (f - T_START) // T_CAND
            ph = (f - T_START) % T_CAND
            draw_highlight(d, k)
            if T_HL <= ph < T_HL + T_PREV:
                a = min(1.0, (ph - T_HL + 1) / 2.0)
                img = preview_overlay(img, CANDS[k]["shape"], a)
        elif T_MOVE_START <= f < T_LAND:
            t = ease((f - T_MOVE_START + 1) / T_MOVE)
            k = next(i for i, c in enumerate(CANDS) if c["match"])
            c = CANDS[k]
            src, dst = c["origin"], GAP_ORIGIN
            for cx, cy in c["shape"]:
                sx = src[0] + C_PITCH * cx; sy = src[1] + C_PITCH * cy
                dx = dst[0] + S_PITCH * cx; dy = dst[1] + S_PITCH * cy
                x = sx + (dx - sx) * t; y = sy + (dy - sy) * t
                s = C_SIZE + (S_SIZE - C_SIZE) * t
                d.rectangle((round(x), round(y), round(x + s), round(y + s)), fill=PURPLE, outline=OUTLINE)
        elif f >= T_LAND:
            img = remove_dashes(img)
            d = ImageDraw.Draw(img)
            draw_cells(d, GAP_ORIGIN, S_PITCH, S_SIZE, GAP_SHAPE)
        frames.append(np.array(img.convert("RGB")))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", "1024x1024", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "10", "-preset", "slow",
           "-r", str(FPS), OUT]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fr in frames:
        p.stdin.write(fr.tobytes())
    p.stdin.close()
    p.wait()
    Image.fromarray(frames[-1]).save(os.path.join(APP, "output", "last_frame.png"))
    print("wrote", OUT, len(frames), "frames")


if __name__ == "__main__":
    main()
