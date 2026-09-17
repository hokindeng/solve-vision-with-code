from pathlib import Path
import subprocess
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path('/app')
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)

def main():
    base = Image.open(ROOT / 'first_frame.png').convert('RGB')
    # Only the purple disk is a circle contained in the circular region.
    # The containing circle itself is a region boundary, not a target object.
    cx, cy, radius = 853.5, 573.5, 25.5
    scale = 4
    box = (820, 540, 887, 607)
    w, h = box[2] - box[0], box[3] - box[1]
    proc = subprocess.Popen([
        'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
        '-pixel_format', 'rgb24', '-video_size', '1024x1024',
        '-framerate', '16', '-i', '-', '-an', '-c:v', 'libx264',
        '-preset', 'slow', '-crf', '0', '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart', str(OUT / 'video.mp4')
    ], stdin=subprocess.PIPE)
    for i in range(40):
        frame = base.copy()
        if i:
            mask = Image.new('L', (w * scale, h * scale), 0)
            draw = ImageDraw.Draw(mask)
            bounds = tuple(round(v * scale) for v in (
                cx - radius - box[0], cy - radius - box[1],
                cx + radius - box[0], cy + radius - box[1]))
            draw.arc(bounds, -90, -90 + 360 * i / 39,
                     fill=255, width=3 * scale)
            mask = mask.resize((w, h), Image.Resampling.LANCZOS)
            frame.paste((0, 175, 40), box, mask)
        proc.stdin.write(np.asarray(frame).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError('ffmpeg failed')

if __name__ == '__main__':
    main()
