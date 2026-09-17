from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import imageio.v2 as imageio

ROOT = Path('/app')

def main():
    base = Image.open(ROOT / 'first_frame.png').convert('RGB')
    a = np.asarray(base)
    ys, xs = np.where(np.all(a == (25, 255, 25), axis=2))
    top = (float(xs[ys == ys.min()].mean()), float(ys.min()))
    bottom_y = ys.max()
    bottom_x = xs[ys == bottom_y]
    vertices = [top, (float(bottom_x.max()), float(bottom_y)),
                (float(bottom_x.min()), float(bottom_y)), top]
    lengths = [np.hypot(q[0]-p[0], q[1]-p[1]) for p,q in zip(vertices, vertices[1:])]
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    with imageio.get_writer(out / 'video.mp4', fps=16, codec='libx264',
                            pixelformat='yuv420p', macro_block_size=1,
                            ffmpeg_params=['-crf', '0', '-preset', 'medium']) as writer:
        for frame in range(40):
            im = base.copy()
            if frame:
                distance = sum(lengths) * frame / 39
                points = [vertices[0]]
                for p, q, length in zip(vertices, vertices[1:], lengths):
                    if distance >= length:
                        points.append(q)
                        distance -= length
                    else:
                        t = distance / length
                        points.append((p[0]+t*(q[0]-p[0]), p[1]+t*(q[1]-p[1])))
                        break
                ImageDraw.Draw(im).line(points, fill=(255, 0, 0), width=6, joint='curve')
            writer.append_data(np.asarray(im))

if __name__ == '__main__':
    main()
