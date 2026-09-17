from pathlib import Path
import math
import subprocess
import numpy as np
from PIL import Image

ROOT = Path('/app')
FPS = 16
FRAMES = 80

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    background = original.copy()
    xs = [240, 484, 728]
    sprites = [original[94:151, x:x+57].copy() for x in xs]
    for x in xs:
        background[94:151, x:x+57] = 255
    surfaces = [496, 456, 590]
    out = ROOT / 'output' / 'video.mp4'
    out.parent.mkdir(exist_ok=True)
    encoder = subprocess.Popen([
        'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
        '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024',
        '-r', str(FPS), '-i', '-', '-an', '-c:v', 'libx264', '-crf', '18',
        '-preset', 'slow', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out)
    ], stdin=subprocess.PIPE)
    for f in range(FRAMES):
        t = f / FPS
        canvas = background.copy()
        for i, (x, surface) in enumerate(zip(xs, surfaces)):
            contact_y = surface - 57
            contact_time = .3 + math.sqrt(2 * (contact_y - 94) / 360)
            if t <= .3:
                y = 94
            elif t < contact_time:
                y = 94 + 180 * (t - .3)**2
            elif i < 2:
                # Drag slows the descent; the base comes to rest on the cup floor.
                u = min(1., (t-contact_time)/(4.65-contact_time))
                ease = 1 - (1-u)**1.5
                y = contact_y + (787-contact_y)*ease
            else:
                # Buoyancy arrests the fall, followed by a damped bob at the surface.
                s = t-contact_time
                equilibrium = 567
                y = equilibrium + math.exp(-2.5*s)*(-34*math.cos(5*s)+25*math.sin(5*s))
                if t > 4.55:
                    blend = min(1., (t-4.55)/.2)
                    y = y*(1-blend)+equilibrium*blend
            y = int(round(y))
            canvas[y:y+57, x:x+57] = sprites[i]
        if f == 0:
            assert np.array_equal(canvas, original)
        encoder.stdin.write(canvas.tobytes())
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError('Video encoding failed')

if __name__ == '__main__':
    main()
