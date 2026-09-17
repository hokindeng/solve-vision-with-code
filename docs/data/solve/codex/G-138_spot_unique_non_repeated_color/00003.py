from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import subprocess

ROOT = Path('/app')

def main():
    base = Image.open(ROOT / 'first_frame.png').convert('RGB')
    a = np.array(base)
    yy, xx = np.where(np.all(a == (165, 236, 85), axis=2))
    x0, x1, y0, y1 = int(xx.min()), int(xx.max()), int(yy.min()), int(yy.max())
    # Draw the contour inside the shape boundary, preserving the background.
    inset = 2
    points = [(x0+inset,y0+inset),(x1-inset,y0+inset),
              (x1-inset,y1-inset),(x0+inset,y1-inset),(x0+inset,y0+inset)]
    lengths = [abs(q[0]-p[0])+abs(q[1]-p[1]) for p,q in zip(points,points[1:])]
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    command = ['ffmpeg','-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24',
               '-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264',
               '-preset','slow','-crf','0','-pix_fmt','yuv420p','-movflags','+faststart',str(out/'video.mp4')]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    for i in range(21):
        frame = base.copy()
        if i:
            draw = ImageDraw.Draw(frame)
            remaining = sum(lengths) * i / 20
            for p,q,length in zip(points,points[1:],lengths):
                if remaining <= 0:
                    break
                ratio = min(remaining / length, 1)
                end = (round(p[0]+(q[0]-p[0])*ratio), round(p[1]+(q[1]-p[1])*ratio))
                draw.line([p,end], fill=(0,0,0), width=5)
                remaining -= length
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    stderr = proc.stderr.read()
    if proc.wait():
        raise RuntimeError(stderr.decode())

if __name__ == '__main__':
    main()
