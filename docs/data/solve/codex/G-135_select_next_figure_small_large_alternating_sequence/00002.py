from pathlib import Path
import subprocess
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path('/app')

def main():
    base = Image.open(ROOT / 'first_frame.png').convert('RGB')
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    proc = subprocess.Popen([
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16', '-i', '-',
        '-an', '-c:v', 'libx264', '-crf', '18', '-preset', 'slow',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out / 'video.mp4')
    ], stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    # Pause for the size pattern to be read, then trace the answer circle.
    scale = 4
    for frame in range(60):
        image = base.copy()
        if frame >= 16:
            progress = min(1.0, (frame - 16) / 31.0)
            if progress > 0:
                overlay = Image.new('RGBA', (1024*scale, 1024*scale))
                draw = ImageDraw.Draw(overlay)
                cx, cy, radius = 626.5, 854.5, 72
                angles = np.linspace(-np.pi/2, -np.pi/2 + 2*np.pi*progress,
                                     max(2, int(500*progress)))
                points = [((cx + radius*np.cos(a))*scale,
                           (cy + radius*np.sin(a))*scale) for a in angles]
                draw.line(points, fill=(230, 30, 35, 255), width=5*scale)
                for x,y in (points[0], points[-1]):
                    r = 2.5*scale
                    draw.ellipse((x-r,y-r,x+r,y+r), fill=(230,30,35,255))
                overlay = overlay.resize(base.size, Image.Resampling.LANCZOS)
                image.paste(overlay, (0,0), overlay)
        proc.stdin.write(image.tobytes())
    proc.stdin.close()
    errors = proc.stderr.read().decode()
    if proc.wait() != 0:
        raise RuntimeError(errors)

if __name__ == '__main__':
    main()
