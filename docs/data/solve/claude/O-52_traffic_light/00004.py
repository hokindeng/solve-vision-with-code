#!/usr/bin/env python3
"""Render the traffic-light simulation video from first_frame.png.

Each light cycles Red(4s) -> Yellow(4s) -> Green(4s) -> Yellow(4s) -> Red.
Assumption: the yellow lights in the first frame are the yellow that follows
red (i.e. they turn green next), matching the order the prompt lists.
The 6-second simulation is paced over the 7-second video with a short hold
on the final state.
"""
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BASE = "/app/first_frame.png"
OUT = "/app/output/video.mp4"
FPS = 16
N_FRAMES = 112
SIM_SECONDS = 6.0
HOLD_FRAMES = 8  # frames holding the final state at the end

FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 102)

RED = (255, 0, 0)
YELLOW = (255, 200, 0)
GREEN = (0, 200, 0)
CYCLE = [("red", 4.0), ("yellow", 4.0), ("green", 4.0), ("yellow", 4.0)]
COLOR = {"red": RED, "yellow": YELLOW, "green": GREEN}

# Light geometry measured from first_frame.png:
#   circle (cx, cy), box interior (x0, y0, x1, y1) inclusive.
LIGHTS = {
    "north": {"circle": (512, 214), "box": (451, 276, 573, 398), "phase": 0, "remain": 4.0},
    "south": {"circle": (512, 798), "box": (451, 860, 573, 982), "phase": 1, "remain": 4.0},
    "east":  {"circle": (804, 506), "box": (743, 568, 865, 690), "phase": 1, "remain": 1.0},
    "west":  {"circle": (220, 506), "box": (159, 568, 281, 690), "phase": 1, "remain": 3.0},
}


def state_at(light, t):
    """Return (color_name, remaining_seconds) for a light at sim time t."""
    phase, remain = light["phase"], light["remain"]
    while t >= remain - 1e-9:
        t -= remain
        phase = (phase + 1) % len(CYCLE)
        remain = CYCLE[phase][1]
    return CYCLE[phase][0], remain - t


def build_masks(base):
    """Per light: boolean mask of the circle fill pixels (exact fill colour)."""
    masks = {}
    for name, L in LIGHTS.items():
        cx, cy = L["circle"]
        yy, xx = np.mgrid[0:base.shape[0], 0:base.shape[1]]
        disc = (xx - cx) ** 2 + (yy - cy) ** 2 <= 70 ** 2
        fill = ((base == RED).all(2) | (base == YELLOW).all(2) | (base == GREEN).all(2))
        masks[name] = disc & fill
    return masks


def render(base, masks, t):
    arr = base.copy()
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)
    for name, L in LIGHTS.items():
        color, remain = state_at(L, t)
        secs = int(np.ceil(remain - 1e-6))
        x0, y0, x1, y1 = L["box"]
        draw.rectangle([x0, y0, x1, y1], fill=(255, 255, 255))
        cx = (x0 + x1) // 2
        cy = (y0 + y1) // 2 + 2
        draw.text((cx, cy), str(secs), font=FONT, fill=(0, 0, 0), anchor="mm")
    out = np.array(img)
    for name, L in LIGHTS.items():
        color, _ = state_at(L, t)
        out[masks[name]] = COLOR[color]
    return out


def main():
    base = np.array(Image.open(BASE).convert("RGB"))
    masks = build_masks(base)
    active = N_FRAMES - HOLD_FRAMES  # frame index `active-1` reaches SIM_SECONDS
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", "1024x1024", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-r", str(FPS), OUT]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(N_FRAMES):
        t = min(i, active - 1) / (active - 1) * SIM_SECONDS
        frame = base if i == 0 else render(base, masks, t)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait()
    print("wrote", OUT)


if __name__ == "__main__":
    main()
