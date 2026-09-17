from pathlib import Path
import math
import subprocess
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path('/app')
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
base = Image.open(ROOT / 'first_frame.png').convert('RGB')
# The three corners of the original triangle, in image coordinates.
vertices = np.array([[267., 478.], [475., 291.], [470., 835.]])
angles = []
for i, vertex in enumerate(vertices):
    u = vertices[(i + 1) % 3] - vertex
    v = vertices[(i + 2) % 3] - vertex
    angles.append(math.acos(np.clip(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)), -1, 1)))
cx, cy = vertices[int(np.argmax(angles))]

command = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
           '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024',
           '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '0',
           '-preset', 'slow', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
           str(OUT / 'video.mp4')]
process = subprocess.Popen(command, stdin=subprocess.PIPE)
for frame_no in range(22):
    frame = base.copy()
    # Brief observation, a smoothly drawn circle, and a final completed hold.
    if frame_no >= 3:
        progress = min(1., (frame_no - 2) / 17.)
        scale = 4
        mask = Image.new('L', (1024 * scale, 1024 * scale), 0)
        draw = ImageDraw.Draw(mask)
        radius = 37
        points = []
        for theta in np.linspace(-math.pi / 2, -math.pi / 2 + 2 * math.pi * progress,
                                 max(2, int(300 * progress))):
            points.append(((cx + radius * math.cos(theta)) * scale,
                           (cy + radius * math.sin(theta)) * scale))
        draw.line(points, fill=255, width=4 * scale, joint='curve')
        for x, y in (points[0], points[-1]):
            draw.ellipse((x - 2 * scale, y - 2 * scale, x + 2 * scale, y + 2 * scale), fill=255)
        mask = mask.resize(base.size, Image.Resampling.LANCZOS)
        frame.paste((230, 20, 30), (0, 0), mask)
    process.stdin.write(frame.tobytes())
process.stdin.close()
if process.wait() != 0:
    raise RuntimeError('Video encoding failed')
print('Interior angles:', [round(math.degrees(a), 2) for a in angles])
print(OUT / 'video.mp4')
