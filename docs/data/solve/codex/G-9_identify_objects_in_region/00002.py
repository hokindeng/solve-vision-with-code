from pathlib import Path
import subprocess
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)

def main():
    base = Image.open(ROOT / 'first_frame.png').convert('RGB')
    # The sole triangle inside the circle, following its original boundary.
    vertices = np.array([[297., 492.], [330., 557.], [265., 557.], [297., 492.]])
    lengths = np.linalg.norm(np.diff(vertices, axis=0), axis=1)
    total = lengths.sum()
    command = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
               '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024',
               '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '0',
               '-preset', 'medium', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
               str(OUT / 'video.mp4')]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    for frame in range(40):
        image = base.copy()
        if frame:
            remaining = total * frame / 39
            points = [tuple(vertices[0])]
            for start, end, length in zip(vertices[:-1], vertices[1:], lengths):
                fraction = min(1., remaining / length)
                points.append(tuple(start + fraction * (end-start)))
                remaining -= length
                if remaining <= 0:
                    break
            ImageDraw.Draw(image).line(points, fill=(0, 170, 0), width=4, joint='curve')
        process.stdin.write(image.tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError('Video encoding failed')

if __name__ == '__main__':
    main()
