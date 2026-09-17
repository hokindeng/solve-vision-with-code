"""Draw a red circle around the arrow pointing in a different direction.

Four arrows sit tangentially around a circle. Three point counterclockwise
along it; the bottom-left arrow points clockwise, so it is the odd one out.
The red circle is drawn progressively as a sweeping arc over 3 s.
"""
import subprocess
import numpy as np
from PIL import Image, ImageDraw

W = H = 1024
FPS = 16
N_FRAMES = 48
SS = 4  # supersampling factor for anti-aliased circle

base = Image.open('/app/first_frame.png').convert('RGB')

# Bounding box of the odd arrow (found via connected components): x 325..478, y 728..810
CX, CY = 401.5, 769.0
R = 105
THICK = 6
RED = (220, 20, 20)


def overlay(sweep_deg):
    """Return RGBA overlay with a red arc of `sweep_deg` degrees starting at top."""
    ov = Image.new('RGBA', (W * SS, H * SS), (0, 0, 0, 0))
    if sweep_deg <= 0:
        return ov.resize((W, H), Image.LANCZOS)
    d = ImageDraw.Draw(ov)
    bbox = [(CX - R) * SS, (CY - R) * SS, (CX + R) * SS, (CY + R) * SS]
    start = -90
    end = start + sweep_deg
    if sweep_deg >= 360:
        d.ellipse(bbox, outline=RED + (255,), width=THICK * SS)
    else:
        d.arc(bbox, start, end, fill=RED + (255,), width=THICK * SS)
        # round caps so the growing arc looks clean
        rc = THICK * SS / 2
        for ang in (start, end):
            a = np.deg2rad(ang)
            px = (CX + R * np.cos(a)) * SS - (THICK * SS / 2) * np.cos(a)
            py = (CY + R * np.sin(a)) * SS - (THICK * SS / 2) * np.sin(a)
            d.ellipse([px - rc, py - rc, px + rc, py + rc], fill=RED + (255,))
    return ov.resize((W, H), Image.LANCZOS)


def ease(t):
    return 0.5 - 0.5 * np.cos(np.pi * t)


frames = []
for i in range(N_FRAMES):
    if i == 0:
        frames.append(base.copy())
        continue
    # sweep grows from frame 1 to frame 40, then holds complete
    t = min(1.0, i / 40.0)
    sweep = 360.0 * ease(t)
    ov = overlay(sweep)
    frame = Image.alpha_composite(base.convert('RGBA'), ov).convert('RGB')
    frames.append(frame)

cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
       '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
       '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', '-preset', 'medium',
       '/app/output/video.mp4']
p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
for f in frames:
    p.stdin.write(np.asarray(f, dtype=np.uint8).tobytes())
p.stdin.close()
p.wait()
print('wrote /app/output/video.mp4', p.returncode)
