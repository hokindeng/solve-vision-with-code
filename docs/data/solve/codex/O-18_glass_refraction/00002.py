from pathlib import Path
import math
import subprocess
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path('/app')
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
source = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
# Remove just the printed angle and its arc; retain the normal and incident ray.
mask = np.zeros(source.shape[:2], dtype=bool)
mask[467:493, 555:651] = True
arc = source[469:480, 488:516]
mask[469:480, 488:516] = (arc.max(axis=2) - arc.min(axis=2) < 8) & (arc.min(axis=2) < 250)
angle = math.asin(1.00 / 1.630 * math.sin(math.radians(30)))
start = np.array([512.0, 513.0])
end = np.array([512.0 + (1023.0-513.0)*math.tan(angle), 1023.0])
frames = 70
cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'slow', '-crf', '0', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(OUT/'video.mp4')]
proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
for i in range(frames):
    frame = source.copy()
    fade = min(1.0, i/12.0)
    frame[mask] = np.round(source[mask]*(1-fade) + 255*fade).astype(np.uint8)
    progress = max(0.0, min(1.0, (i-10)/(frames-1-10)))
    if progress:
        tip = start + (end-start)*progress
        # Supersample only the new ray, leaving all other source pixels intact.
        scale = 3
        layer = Image.new('L', (1024*scale, 1024*scale))
        draw = ImageDraw.Draw(layer)
        draw.line([tuple(start*scale), tuple(tip*scale)], fill=255, width=7)
        if np.linalg.norm(tip-start) > 20:
            direction = np.array([math.sin(angle), math.cos(angle)])
            perpendicular = np.array([-direction[1], direction[0]])
            for sign in [-1, 1]:
                tail = tip - direction*14 + sign*perpendicular*5
                draw.line([tuple(tail*scale), tuple(tip*scale)], fill=255, width=7)
        alpha = np.array(layer.resize((1024,1024),Image.Resampling.LANCZOS)).astype(float)/255
        alpha[:514] = 0
        frame = np.round(frame*(1-alpha[:,:,None]) + np.array([255,0,0])*alpha[:,:,None]).astype(np.uint8)
    proc.stdin.write(frame.tobytes())
    if i == frames-1:
        Image.fromarray(frame).save(OUT/'last_frame.png')
proc.stdin.close()
if proc.wait():
    raise RuntimeError('ffmpeg failed')
print(f'Refraction angle: {math.degrees(angle):.6f} degrees; boundary x={end[0]:.3f}')
