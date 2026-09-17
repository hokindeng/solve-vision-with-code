from pathlib import Path
import subprocess
import numpy as np
from PIL import Image

ROOT = Path('/app')
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
base = Image.open(ROOT / 'first_frame.png').convert('RGB')
# The outlines span 150 pixels in A and 130 pixels in B.
scale = 130 / 150
source = base.crop((180, 693, 331, 844))
question_box = (750, 740, 790, 796)
question = base.crop(question_box)
white_question = Image.new('RGB', question.size, 'white')

def ease(t):
    t = max(0.0, min(1.0, t))
    return t*t*(3-2*t)

def frame(i):
    if i == 0:
        return base.copy()
    canvas = base.copy()
    canvas.paste(Image.blend(question, white_question, ease(i/16)), question_box[:2])
    alpha = ease((i-16)/12)
    if alpha:
        progress = ease((i-22)/35)
        extent = 150 * (1 - (1-scale)*progress)
        side = round(extent) + 1
        shape = source.resize((side, side), Image.Resampling.LANCZOS)
        shape = Image.blend(Image.new('RGB', shape.size, 'white'), shape, alpha)
        canvas.paste(shape, (769-side//2, 768-side//2))
    return canvas

command = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
           '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024',
           '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '0',
           '-preset', 'medium', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
           str(OUT / 'video.mp4')]
with subprocess.Popen(command, stdin=subprocess.PIPE) as proc:
    for i in range(60):
        img = frame(i)
        # All changes are restricted to the missing answer area.
        a, b = np.asarray(img), np.asarray(base)
        mask = np.any(a != b, axis=2)
        mask[690:847, 690:847] = False
        assert not mask.any()
        proc.stdin.write(a.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError('ffmpeg failed')
