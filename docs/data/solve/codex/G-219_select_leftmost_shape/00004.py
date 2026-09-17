from pathlib import Path
import subprocess
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent

def main():
    base = Image.open(ROOT / 'first_frame.png').convert('RGB')
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    # A circle centered on the leftmost triangle's circumcenter, with clearance.
    cx, cy, radius = 340.5, 368.0, 64.0
    scale = 4
    proc = subprocess.Popen([
        'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
        '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16', '-i', '-',
        '-an', '-c:v', 'libx264', '-preset', 'slow', '-crf', '10',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out / 'video.mp4')
    ], stdin=subprocess.PIPE)
    for i in range(48):
        frame = base.copy()
        if i:
            overlay = Image.new('RGBA', (1024*scale, 1024*scale))
            draw = ImageDraw.Draw(overlay)
            progress = i / 47
            angles = np.linspace(-np.pi/2, -np.pi/2 + 2*np.pi*progress, max(2, int(500*progress)))
            points = [((cx+radius*np.cos(a))*scale, (cy+radius*np.sin(a))*scale) for a in angles]
            draw.line(points, fill=(255, 0, 0, 255), width=5*scale, joint='curve')
            for x,y in (points[0], points[-1]):
                r=2.5*scale
                draw.ellipse((x-r,y-r,x+r,y+r), fill=(255,0,0,255))
            overlay = overlay.resize(base.size, Image.Resampling.LANCZOS)
            frame.paste(overlay, (0,0), overlay)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait():
        raise RuntimeError('ffmpeg failed')

if __name__ == '__main__':
    main()
