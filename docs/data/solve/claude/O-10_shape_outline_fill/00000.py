import math, os, subprocess
import numpy as np
from PIL import Image, ImageDraw

BASE = Image.open('/app/first_frame.png').convert('RGB')
W, H = BASE.size
FPS, N = 16, 60
GREEN = (70, 153, 53)
WHITE = (255, 255, 255)

# Row 1: outline width 4 -> 2 (thick -> thin). Apply same to the octagon.
C_CENTER, R = (160, 768), 80
ANS_CENTER = (864, 768)
W_START, W_END = 4.0, 2.0
QBOX = (845, 740, 885, 795)  # region containing the "?" placeholder

def octagon(center, r, scale=1):
    cx, cy = center
    pts = [(round(cx + r * math.cos(math.radians(-90 + 45 * k))),
            round(cy + r * math.sin(math.radians(-90 + 45 * k)))) for k in range(8)]
    pts = [(x * scale, y * scale) for x, y in pts]
    return pts + [pts[0]]

def draw_octagon_layer(center, width, alpha, ss=4):
    """Return RGBA layer (full size) with the octagon outline drawn at fractional width."""
    if abs(width - round(width)) < 1e-6 and ss == 1:
        img = Image.new('RGBA', (W, H), (0, 0, 0, 0))
        ImageDraw.Draw(img).line(octagon(center, R), fill=GREEN + (255,), width=int(round(width)))
    else:
        img = Image.new('RGBA', (W * ss, H * ss), (0, 0, 0, 0))
        ImageDraw.Draw(img).line(octagon(center, R, ss), fill=GREEN + (255,),
                                 width=max(1, int(round(width * ss))))
        img = img.resize((W, H), Image.LANCZOS)
    if alpha < 1:
        a = img.split()[3].point(lambda v: int(v * alpha))
        img.putalpha(a)
    return img

def ease(t):
    return 0.5 - 0.5 * math.cos(math.pi * min(max(t, 0.0), 1.0))

def frame(i):
    t = i / (N - 1)
    img = BASE.copy()
    if i == 0:
        return img
    # Phase 1 (first 25%): "?" fades out, octagon (style of C) fades in.
    p1 = ease(t / 0.25)
    if p1 > 0:
        q = img.crop(QBOX)
        white = Image.new('RGB', q.size, WHITE)
        img.paste(Image.blend(q, white, p1), QBOX[:2])
    # Phase 2 (remaining 75%): outline thins from 4 to 2, like A -> B.
    p2 = ease((t - 0.25) / 0.75)
    width = W_START + (W_END - W_START) * p2
    layer = draw_octagon_layer(ANS_CENTER, width, p1, ss=4)
    img.paste(layer, (0, 0), layer)
    return img

def main():
    os.makedirs('/app/output', exist_ok=True)
    os.makedirs('/app/frames', exist_ok=True)
    for i in range(N):
        frame(i).save(f'/app/frames/{i:03d}.png')
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(FPS),
                    '-i', '/app/frames/%03d.png', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                    '-crf', '12', '-r', str(FPS), '/app/output/video.mp4'], check=True)

if __name__ == '__main__':
    main()
