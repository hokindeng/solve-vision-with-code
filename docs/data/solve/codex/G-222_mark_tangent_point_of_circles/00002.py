from pathlib import Path
import math
import numpy as np
from PIL import Image, ImageDraw
import subprocess

ROOT = Path('/app')

def main():
    base = Image.open(ROOT / 'first_frame.png').convert('RGB')
    arr = np.asarray(base)
    centers = []
    radii = []
    for color in [(96, 125, 139), (22, 160, 133)]:
        yy, xx = np.where(np.all(arr == color, axis=2))
        centers.append(np.array([(xx.min()+xx.max())/2, (yy.min()+yy.max())/2]))
        radii.append((xx.max()-xx.min()+1)/2)
    direction = centers[1] - centers[0]
    tangent = centers[0] + direction * radii[0] / np.linalg.norm(direction)
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    encoder = subprocess.Popen([
        'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', '1024x1024', '-r', '16', '-i', '-', '-an', '-c:v', 'libx264',
        '-preset', 'slow', '-crf', '15', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
        str(out / 'video.mp4')
    ], stdin=subprocess.PIPE)
    scale = 4
    for frame in range(60):
        result = base.copy()
        progress = max(0.0, min(1.0, (frame-3)/52))
        if progress > 0:
            mask = Image.new('L', (1024*scale, 1024*scale), 0)
            draw = ImageDraw.Draw(mask)
            radius = 23
            angles = np.linspace(-math.pi/2, -math.pi/2 + progress*2*math.pi, max(2, int(progress*360)))
            points = [((tangent[0]+radius*math.cos(a))*scale, (tangent[1]+radius*math.sin(a))*scale) for a in angles]
            draw.line(points, fill=255, width=4*scale, joint='curve')
            for x,y in (points[0], points[-1]):
                draw.ellipse((x-2*scale,y-2*scale,x+2*scale,y+2*scale), fill=255)
            mask = mask.resize(base.size, Image.Resampling.LANCZOS)
            result.paste((0,0,0), mask=mask)
        encoder.stdin.write(result.tobytes())
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError('Video encoding failed')

if __name__ == '__main__':
    main()
