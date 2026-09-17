from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
import subprocess

ROOT = Path(__file__).resolve().parent

def frame(t):
    # The original outlines match Pillow's integer ellipse rasterization exactly.
    im = Image.new('RGB', (1024, 1024), (255, 255, 255))
    draw = ImageDraw.Draw(im)
    ease = t*t*(3-2*t)
    for x, y, radius, color in [(437,307,423,(128,128,0)), (910,716,100,(245,130,48))]:
        cx = round(x + (512-x)*ease)
        cy = round(y + (512-y)*ease)
        draw.ellipse((cx-radius,cy-radius,cx+radius,cy+radius), outline=color, width=8)
    return np.asarray(im)

def main():
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    original = np.asarray(Image.open(ROOT/'first_frame.png').convert('RGB'))
    assert np.array_equal(frame(0), original)
    command = ['ffmpeg','-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24',
               '-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264',
               '-preset','slow','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(out/'video.mp4')]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE)
    for i in range(40):
        proc.stdin.write(frame(i/39).tobytes())
    proc.stdin.close()
    if proc.wait():
        raise RuntimeError('ffmpeg encoding failed')

if __name__ == '__main__':
    main()
