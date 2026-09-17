from pathlib import Path
import math
import subprocess
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
source = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
height, width = source.shape[:2]
# Remove only the black angle label and arc. Preserve the incident blue ray
# and the fine vertical normal through the point of incidence.
annotation = np.zeros((height, width), dtype=bool)
annotation[418:445, 730:825] = True
arc_box = np.zeros_like(annotation)
arc_box[423:433, 663:688] = True
neutral = (source.max(axis=2).astype(int) - source.min(axis=2).astype(int)) < 12
annotation |= arc_box & neutral & (source.min(axis=2) < 245)
annotation &= np.any(source != 255, axis=2)

origin = np.array([685., 465.])
# Reflect the incident direction across the horizontal mirror.
direction = np.array([685. - 441., -(465. - 50.)])
end = origin + direction * (origin[1] / -direction[1])
unit = direction / np.linalg.norm(direction)
perp = np.array([-unit[1], unit[0]])
reflectivity = 0.46
color = tuple(round(255 * (1 - reflectivity) + c * reflectivity) for c in (0, 0, 255))

process = subprocess.Popen([
    'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
    '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{width}x{height}',
    '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '0',
    '-preset', 'slow', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
    str(OUT / 'video.mp4')
], stdin=subprocess.PIPE)
for frame in range(35):
    pixels = source.copy()
    if frame:
        fade = min(frame / 6., 1.)
        pixels[annotation] = np.rint(source[annotation] * (1 - fade) + 255 * fade).astype(np.uint8)
        progress = max(0., min((frame - 3) / 31., 1.))
        if progress > 0:
            # Supersampling smooths the new ray without resampling the original.
            scale = 4
            mask_image = Image.new('L', (width * scale, height * scale))
            draw = ImageDraw.Draw(mask_image)
            tip = origin + (end - origin) * progress
            def point(p):
                return tuple(float(v * scale) for v in p)
            draw.line([point(origin), point(tip)], fill=255, width=2 * scale)
            if np.linalg.norm(tip - origin) > 32:
                back = tip - unit * 22
                draw.line([point(back + perp * 8), point(tip), point(back - perp * 8)], fill=255, width=2 * scale)
            mask = np.asarray(mask_image.resize((width, height), Image.Resampling.LANCZOS)).astype(float) / 255.
            # Keep the mirror surface and everything below it exactly intact.
            mask[462:, :] = 0
            alpha = mask[:, :, None]
            pixels = np.rint(pixels * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
    process.stdin.write(pixels.tobytes())
process.stdin.close()
if process.wait() != 0:
    raise RuntimeError('ffmpeg failed')
