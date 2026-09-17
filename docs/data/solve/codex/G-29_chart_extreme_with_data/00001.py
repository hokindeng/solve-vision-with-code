from pathlib import Path
import subprocess
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path('/app')

def main():
    base = Image.open(ROOT / 'first_frame.png').convert('RGB')
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    # Enclose the maximum-valued marker (97) and its data label.
    corners = [(283, 186), (342, 186), (342, 272), (283, 272), (283, 186)]
    lengths = [abs(b[0]-a[0])+abs(b[1]-a[1]) for a,b in zip(corners, corners[1:])]
    encoder = subprocess.Popen([
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16', '-i', '-',
        '-an', '-c:v', 'libx264', '-crf', '15', '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart', str(out / 'video.mp4')
    ], stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    for frame in range(48):
        image = base.copy()
        if frame > 0:
            draw = ImageDraw.Draw(image)
            distance = sum(lengths) * min(frame / 45, 1)
            for a, b, length in zip(corners, corners[1:], lengths):
                if distance <= 0:
                    break
                fraction = min(distance / length, 1)
                endpoint = (round(a[0] + (b[0]-a[0])*fraction),
                            round(a[1] + (b[1]-a[1])*fraction))
                draw.line([a, endpoint], fill=(255, 0, 0), width=4)
                distance -= length
        encoder.stdin.write(np.asarray(image).tobytes())
    encoder.stdin.close()
    error = encoder.stderr.read()
    if encoder.wait() != 0:
        raise RuntimeError(error.decode())

if __name__ == '__main__':
    main()
