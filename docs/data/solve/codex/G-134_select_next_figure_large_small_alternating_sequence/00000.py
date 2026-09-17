from pathlib import Path
import math
import numpy as np
from PIL import Image, ImageDraw
import imageio.v2 as imageio

ROOT = Path('/app')
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
BASE = Image.open(ROOT / 'first_frame.png').convert('RGB')
S = 4
RED = (225, 35, 40, 255)
TEAL = (0, 128, 128, 255)

def arc(draw, cx, cy, radius, progress, width=4):
    box = tuple(round(v*S) for v in (cx-radius, cy-radius, cx+radius, cy+radius))
    draw.arc(box, -90, -90+359.9*max(0,min(1,progress)), fill=RED, width=width*S)

def frame(i):
    if i == 0:
        return np.asarray(BASE)
    layer = Image.new('RGBA', (1024*S, 1024*S))
    d = ImageDraw.Draw(layer)
    # Read each size in order, using a temporary tracing ring.
    for start, end, cx, radius in [(3,13,241.5,42), (14,24,421.5,63), (25,35,602.5,42)]:
        if start <= i <= end:
            arc(d, cx, 337.5, radius, (i-start+1)/7)
    # Demonstrate the predicted large circle inside the missing-item box.
    if i >= 36:
        progress = min(1, (i-35)/9)
        r = 54.5 * progress
        # The growing circle covers the question mark as the answer appears.
        d.ellipse(tuple(round(v*S) for v in (782-r,337.5-r,782+r,337.5+r)), fill=TEAL)
    # Select the matching large teal circle, and hold the finished result.
    if i >= 46:
        arc(d, 856.5, 854.5, 76, (i-45)/10, 5)
    layer = layer.resize(BASE.size, Image.Resampling.LANCZOS)
    return np.asarray(Image.alpha_composite(BASE.convert('RGBA'), layer).convert('RGB'))

with imageio.get_writer(OUT / 'video.mp4', fps=16, codec='libx264', pixelformat='yuv420p', macro_block_size=None, ffmpeg_params=['-crf','18']) as writer:
    for i in range(60):
        writer.append_data(frame(i))
