from pathlib import Path
import subprocess
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
base = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
shape = np.all(base == (25, 25, 255), axis=2)
vertices = np.array([[520., 387.], [680., 665.], [360., 665.], [520., 387.]])
lengths = np.linalg.norm(np.diff(vertices, axis=0), axis=1)
scale = 4
process = subprocess.Popen([
    'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
    '-pixel_format', 'rgb24', '-video_size', '1024x1024', '-framerate', '16',
    '-i', '-', '-an', '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p',
    '-movflags', '+faststart', str(OUT / 'video.mp4')
], stdin=subprocess.PIPE)
for frame_index in range(40):
    frame = base.copy()
    if frame_index:
        remaining = lengths.sum() * frame_index / 39
        mask = Image.new('L', (1024*scale, 1024*scale))
        draw = ImageDraw.Draw(mask)
        for a, b, length in zip(vertices[:-1], vertices[1:], lengths):
            if remaining <= 0:
                break
            end = a + (b-a)*min(remaining/length, 1)
            draw.line([tuple(a*scale), tuple(end*scale)], fill=255, width=12*scale)
            remaining -= length
        alpha = np.asarray(mask.resize((1024,1024), Image.Resampling.LANCZOS)).astype(float)/255
        alpha *= shape
        frame = np.rint(base*(1-alpha[:,:,None]) + np.array([255,0,0])*alpha[:,:,None]).astype(np.uint8)
    process.stdin.write(frame.tobytes())
process.stdin.close()
if process.wait():
    raise RuntimeError('ffmpeg failed')
