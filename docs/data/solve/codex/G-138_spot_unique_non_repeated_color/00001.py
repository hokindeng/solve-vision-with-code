from pathlib import Path
import subprocess
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent

def main():
    original = Image.open(ROOT / 'first_frame.png').convert('RGB')
    a = np.array(original)
    target = np.all(a == (210, 91, 236), axis=2)
    ys, xs = np.where(target)
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    # Keep the contour entirely within the original rectangle.
    inset = 2
    vertices = [(x0+inset,y0+inset), (x1-inset,y0+inset),
                (x1-inset,y1-inset), (x0+inset,y1-inset), (x0+inset,y0+inset)]
    lengths = [abs(q[0]-p[0])+abs(q[1]-p[1]) for p,q in zip(vertices,vertices[1:])]
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    cmd = ['ffmpeg','-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24',
           '-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264',
           '-crf','0','-preset','slow','-pix_fmt','yuv420p','-movflags','+faststart',str(out/'video.mp4')]
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    for frame in range(21):
        result = a.copy()
        if frame:
            remaining = sum(lengths) * frame / 20
            points = [vertices[0]]
            for p, q, length in zip(vertices, vertices[1:], lengths):
                distance = min(length, remaining)
                t = distance / length
                points.append((round(p[0]+(q[0]-p[0])*t), round(p[1]+(q[1]-p[1])*t)))
                remaining -= distance
                if remaining <= 0:
                    break
            mask = Image.new('L', original.size)
            ImageDraw.Draw(mask).line(points, fill=255, width=5, joint='curve')
            result[(np.array(mask)>0) & target] = 0
        process.stdin.write(result.tobytes())
    process.stdin.close()
    error = process.stderr.read()
    if process.wait():
        raise RuntimeError(error.decode())

if __name__ == '__main__':
    main()
