from pathlib import Path
import math
import numpy as np
from PIL import Image, ImageDraw
import imageio.v2 as imageio

ROOT = Path('/app')
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
base = Image.open(ROOT / 'first_frame.png').convert('RGB')
# The values are 32, 92, 96, and 51; only 96 receives a circle.
# Hold the original while comparing, then trace one continuous circle.
scale = 3
center = (716, 250)
radius = 110
with imageio.get_writer(str(OUT / 'video.mp4'), fps=16, codec='libx264',
                        pixelformat='yuv420p', quality=10,
                        macro_block_size=1, ffmpeg_params=['-crf', '18']) as writer:
    for frame in range(80):
        progress = max(0.0, min(1.0, (frame - 12) / 59))
        canvas = base.copy()
        if progress > 0:
            overlay = Image.new('RGBA', (base.width * scale, base.height * scale))
            draw = ImageDraw.Draw(overlay)
            count = max(2, int(720 * progress))
            points = []
            for k in range(count + 1):
                angle = -math.pi / 2 + 2 * math.pi * progress * k / count
                points.append(((center[0] + radius * math.cos(angle)) * scale,
                               (center[1] + radius * math.sin(angle)) * scale))
            color = (230, 20, 25, 255)
            draw.line(points, fill=color, width=6 * scale, joint='curve')
            for x, y in (points[0], points[-1]):
                r = 3 * scale
                draw.ellipse((x-r, y-r, x+r, y+r), fill=color)
            overlay = overlay.resize(base.size, Image.Resampling.LANCZOS)
            canvas.paste(overlay, (0, 0), overlay)
        writer.append_data(np.asarray(canvas))
