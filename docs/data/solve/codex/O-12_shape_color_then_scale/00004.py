from pathlib import Path
import numpy as np
from PIL import Image
import subprocess

ROOT = Path('/app')
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
base = Image.open(ROOT / 'first_frame.png').convert('RGB')
# Reuse the exact source silhouette and its black border.
oval = np.array(base.crop((82, 633, 279, 732)))
oval[(oval == (153, 132, 30)).all(axis=2)] = (66, 191, 149)
oval = Image.fromarray(oval)

def ease(t):
    t = max(0., min(1., t))
    return t*t*(3-2*t)

def stage(image, center, progress, scale=1.):
    if progress <= 0:
        return image
    # This local area contains only the answer placeholder and white background.
    box = (center-100, 632, center+101, 734)
    before = image.crop(box)
    after = Image.new('RGB', before.size, 'white')
    shape = oval if scale == 1 else oval.resize(
        (round(197*scale), round(99*scale)), Image.Resampling.LANCZOS)
    after.paste(shape, (100-shape.width//2, 50-shape.height//2))
    image.paste(Image.blend(before, after, progress), box)
    return image

command = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
           '-pixel_format', 'rgb24', '-video_size', '1024x1024', '-framerate', '16',
           '-i', '-', '-an', '-c:v', 'libx264', '-crf', '15', '-preset', 'slow',
           '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(OUT/'video.mp4')]
proc = subprocess.Popen(command, stdin=subprocess.PIPE)
for frame in range(60):
    image = base.copy()
    image = stage(image, 518, ease((frame-5)/22))
    reveal = ease((frame-31)/9)
    shrink = ease((frame-39)/16)
    image = stage(image, 854, reveal, 1-.5*shrink)
    proc.stdin.write(np.array(image).tobytes())
proc.stdin.close()
if proc.wait() != 0:
    raise RuntimeError('Video encoding failed')
