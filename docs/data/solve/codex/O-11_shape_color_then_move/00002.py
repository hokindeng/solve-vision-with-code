from pathlib import Path
import numpy as np
from PIL import Image
import subprocess

ROOT = Path('/app')
BASE = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
GREEN = np.array([210, 229, 114], dtype=float)
GOLD = np.array([229, 168, 45], dtype=float)
BAR = BASE[662:703, 115:276].copy()
FILL = np.all(BAR == GREEN, axis=2)

def ease(frame, start, end):
    t = np.clip((frame-start)/(end-start), 0, 1)
    return t*t*(3-2*t)

def remove_question(canvas, x, amount):
    region = BASE[658:706, x:x+32]
    canvas[658:706, x:x+32] = np.rint(region*(1-amount)+255*amount).astype(np.uint8)

def put_bar(canvas, x, y, color, opacity):
    bar = BAR.copy()
    bar[FILL] = np.rint(color).astype(np.uint8)
    old = canvas[y:y+41, x:x+161]
    canvas[y:y+41, x:x+161] = np.rint(old*(1-opacity)+bar*opacity).astype(np.uint8)

def render(i):
    out = BASE.copy()
    reveal = ease(i, 5, 13)
    remove_question(out, 466, reveal)
    color = GREEN + (GOLD-GREEN)*ease(i, 14, 29)
    put_bar(out, 402, 662, color, reveal)
    reveal2 = ease(i, 31, 38)
    remove_question(out, 753, reveal2)
    y = 662 - round(60*ease(i, 39, 54))
    put_bar(out, 689, y, GOLD, reveal2)
    return out

def main():
    (ROOT/'output').mkdir(exist_ok=True)
    cmd = ['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24',
           '-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264',
           '-crf','15','-preset','slow','-pix_fmt','yuv420p','-movflags','+faststart',
           str(ROOT/'output/video.mp4')]
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(60):
        frame = render(i)
        if i == 0:
            assert np.array_equal(frame, BASE)
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError('ffmpeg failed')

if __name__ == '__main__':
    main()
