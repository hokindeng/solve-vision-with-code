#!/usr/bin/env python3
"""Generate /app/output/video.mp4 from /app/first_frame.png.

Scene: a blue block structure with a T-shaped gap (dashed red outline) and four
candidate pieces in boxes below.  The video examines each candidate left to
right (highlight frame -> ghost preview in the gap -> red X / green check), then
flies the matching piece into the gap to complete the structure.
"""
import os
import subprocess
import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")

W = H = 1024
FPS = 16
N_FRAMES = 99

# ---- measured geometry -----------------------------------------------------
BG = (245, 245, 245)
CELL_FILL = (40, 80, 150)
CELL_EDGE = (50, 100, 140)
GRID_X0, GRID_Y0, GRID_PITCH, GRID_CELL = 330, 175, 52, 51      # main structure
BOX_X0, BOX_Y0, BOX_SIZE, BOX_STEP = 50, 829, 186, 246          # candidate boxes
CAND_PITCH, CAND_CELL = 38, 37                                  # candidate cells
BOX_FILL = 230
BOX_EDGE = 180

GREEN = (40, 170, 80)
RED = (220, 60, 60)
AMBER = (255, 160, 30)
SS = 4  # supersampling for anti-aliased overlays


def is_blue(px):
    return abs(int(px[0]) - 40) < 40 and abs(int(px[1]) - 80) < 40 and int(px[2]) > 120


def detect_gap(base):
    """Return set of (row, col) grid cells outlined by the red dashed line."""
    red = (base[:, :, 0] > 200) & (base[:, :, 1] < 150) & (base[:, :, 2] < 150)
    cells = set()
    for r in range(20):
        for c in range(20):
            x = GRID_X0 + c * GRID_PITCH
            y = GRID_Y0 + r * GRID_PITCH
            if x + 52 > W or y + 52 > H:
                continue
            # dashed rectangle occupies [x, x+51] x [y, y+51]; check its top edge row
            top = red[y:y + 2, x:x + 52].sum()
            left = red[y:y + 52, x:x + 2].sum()
            if top > 10 and left > 10 and not is_blue(base[y + 25, x + 25]):
                cells.add((r, c))
    return cells


def detect_structure(base):
    cells = set()
    for r in range(20):
        for c in range(20):
            x = GRID_X0 + c * GRID_PITCH + 25
            y = GRID_Y0 + r * GRID_PITCH + 25
            if x < W and y < H and is_blue(base[y, x]):
                cells.add((r, c))
    return cells


def detect_candidates(base):
    """For each box return (box_rect, [(row, col)...] normalized, pixel origin of cell (0,0))."""
    out = []
    for i in range(4):
        bx = BOX_X0 + i * BOX_STEP
        by = BOX_Y0
        sub = base[by:by + BOX_SIZE, bx:bx + BOX_SIZE]
        blue = (np.abs(sub[:, :, 0].astype(int) - 40) < 40) & (sub[:, :, 2] > 120)
        ys, xs = np.where(blue)
        ox, oy = bx + xs.min(), by + ys.min()
        cells = []
        for r in range(4):
            for c in range(4):
                cx = ox + c * CAND_PITCH + CAND_CELL // 2
                cy = oy + r * CAND_PITCH + CAND_CELL // 2
                if cx < bx + BOX_SIZE and cy < by + BOX_SIZE and is_blue(base[cy, cx]):
                    cells.append((r, c))
        out.append(((bx, by, bx + BOX_SIZE - 1, by + BOX_SIZE - 1), cells, (ox, oy)))
    return out


# ---- drawing helpers -------------------------------------------------------
def draw_cell(draw, x, y, size, fill=CELL_FILL, edge=CELL_EDGE, scale=1):
    x0, y0 = round(x * scale), round(y * scale)
    x1, y1 = round((x + size - 1) * scale + (scale - 1)), round((y + size - 1) * scale + (scale - 1))
    draw.rectangle([x0, y0, x1, y1], fill=fill, outline=edge, width=scale)


def overlay_rgba(img, layer):
    """Composite a supersampled RGBA layer onto the RGB image."""
    small = layer.resize((W, H), Image.LANCZOS)
    img.alpha_composite(small)


def tint_box(arr, rect, tint):
    """Multiply grayscale pixels of the box (fill, border, label) by a colour tint."""
    x0, y0, x1, y1 = rect
    sub = arr[y0:y1 + 1, x0:x1 + 1].astype(np.float32)
    gray = (np.abs(sub[:, :, 0] - sub[:, :, 1]) < 6) & (np.abs(sub[:, :, 1] - sub[:, :, 2]) < 6)
    t = np.array(tint, dtype=np.float32) / 255.0
    tinted = np.clip(sub * t, 0, 255)
    sub[gray] = tinted[gray]
    arr[y0:y1 + 1, x0:x1 + 1] = sub.astype(np.uint8)


def draw_mark(layer_draw, rect, ok):
    """Green check or red cross in the top-right corner of the box (supersampled coords)."""
    x0, y0, x1, y1 = rect
    cx, cy = x1 - 22, y0 + 22
    s = 11
    wdt = 4 * SS
    if ok:
        pts = [(cx - s, cy), (cx - s * 0.3, cy + s * 0.7), (cx + s, cy - s * 0.8)]
        pts = [(p[0] * SS, p[1] * SS) for p in pts]
        layer_draw.line(pts, fill=GREEN + (255,), width=wdt, joint="curve")
    else:
        for (ax, ay, bx, by) in [(-s, -s, s, s), (-s, s, s, -s)]:
            layer_draw.line([((cx + ax) * SS, (cy + ay) * SS), ((cx + bx) * SS, (cy + by) * SS)],
                            fill=RED + (255,), width=wdt)


def draw_highlight(layer_draw, rect):
    x0, y0, x1, y1 = rect
    pad, wdt = 4, 3
    layer_draw.rectangle([(x0 - pad) * SS, (y0 - pad) * SS, (x1 + pad) * SS + SS - 1, (y1 + pad) * SS + SS - 1],
                         outline=AMBER + (255,), width=wdt * SS)


def ease(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def main():
    base = np.array(Image.open(SRC).convert("RGB"))
    gap = detect_gap(base)
    structure = detect_structure(base)
    cands = detect_candidates(base)

    gap_r0 = min(r for r, c in gap)
    gap_c0 = min(c for r, c in gap)
    gap_norm = {(r - gap_r0, c - gap_c0) for r, c in gap}
    match_idx = [i for i, (_, cells, _) in enumerate(cands) if set(cells) == gap_norm]
    assert len(match_idx) == 1, match_idx
    match_idx = match_idx[0]

    # base with the gap completed (final state)
    done = base.copy()
    gx0 = GRID_X0 + gap_c0 * GRID_PITCH
    gy0 = GRID_Y0 + gap_r0 * GRID_PITCH
    gx1 = GRID_X0 + (max(c for r, c in gap) + 1) * GRID_PITCH
    gy1 = GRID_Y0 + (max(r for r, c in gap) + 1) * GRID_PITCH
    reg = done[gy0:gy1 + 1, gx0:gx1 + 1]
    redm = (reg[:, :, 0] > 200) & (reg[:, :, 1] < 150)
    reg[redm] = BG
    done_img = Image.fromarray(done)
    dd = ImageDraw.Draw(done_img)
    for (r, c) in gap:
        draw_cell(dd, GRID_X0 + c * GRID_PITCH, GRID_Y0 + r * GRID_PITCH, GRID_CELL)
    done = np.array(done_img)

    # ---- timeline ----------------------------------------------------------
    PER = 17          # frames per candidate
    T_HL, T_PREV = 4, 7  # highlight only, then preview, remainder judged
    START = 1
    fly_start = START + 4 * PER   # 69
    fly_len = 24
    settle = fly_start + fly_len  # 93 .. 98 completed

    judged = {}  # idx -> bool
    frames = []
    for f in range(N_FRAMES):
        if f >= settle:
            arr = done.copy()
            for i in range(4):
                judged[i] = (i == match_idx)
        else:
            arr = base.copy()
        # which candidates are already judged
        cur = None
        phase = None
        for i in range(4):
            s = START + i * PER
            if f >= s + PER:
                judged[i] = (i == match_idx)
            elif f >= s:
                cur = i
                k = f - s
                phase = "hl" if k < T_HL else ("prev" if k < T_HL + T_PREV else "judge")
                if phase == "judge":
                    judged[i] = (i == match_idx)

        # box tints for judged candidates
        for i, ok in judged.items():
            tint_box(arr, cands[i][0], (205, 245, 215) if ok else (250, 205, 205))

        img = Image.fromarray(arr).convert("RGBA")
        layer = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
        ld = ImageDraw.Draw(layer)

        for i, ok in judged.items():
            draw_mark(ld, cands[i][0], ok)

        if cur is not None:
            draw_highlight(ld, cands[cur][0])
            if phase in ("prev", "judge"):
                # ghost preview: candidate placed with its bbox top-left on the gap bbox top-left
                cells = cands[cur][1]
                for (r, c) in cells:
                    R, C = gap_r0 + r, gap_c0 + c
                    fits = (R, C) in gap
                    col = (40, 80, 150, 150) if fits else (230, 50, 50, 170)
                    edge = (50, 100, 140, 200) if fits else (200, 30, 30, 220)
                    x = GRID_X0 + C * GRID_PITCH
                    y = GRID_Y0 + R * GRID_PITCH
                    ld.rectangle([x * SS, y * SS, (x + GRID_CELL) * SS - 1, (y + GRID_CELL) * SS - 1],
                                 fill=col, outline=edge, width=SS)

        if fly_start <= f < settle:
            # fly a copy of the matching piece from its box to the gap
            t = ease((f - fly_start) / (fly_len - 1))
            rect, cells, (ox, oy) = cands[match_idx]
            size = CAND_CELL + (GRID_CELL - CAND_CELL) * t
            pitch = CAND_PITCH + (GRID_PITCH - CAND_PITCH) * t
            sx = ox + (gx0 - ox) * t
            sy = oy + (gy0 - oy) * t
            for (r, c) in cells:
                x = sx + c * pitch
                y = sy + r * pitch
                ld.rectangle([round(x * SS), round(y * SS), round((x + size) * SS) - 1, round((y + size) * SS) - 1],
                             fill=CELL_FILL + (255,), outline=CELL_EDGE + (255,), width=SS)

        overlay_rgba(img, layer)
        frames.append(np.array(img.convert("RGB")))

    frames[0] = base.copy()  # first frame identical to the source

    os.makedirs(OUT_DIR, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", "-preset", "medium",
           "-movflags", "+faststart", OUT]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fr in frames:
        p.stdin.write(np.ascontiguousarray(fr, dtype=np.uint8).tobytes())
    p.stdin.close()
    p.wait()
    assert p.returncode == 0
    print(f"wrote {OUT}: {len(frames)} frames, match = candidate {match_idx + 1}")


if __name__ == "__main__":
    main()
