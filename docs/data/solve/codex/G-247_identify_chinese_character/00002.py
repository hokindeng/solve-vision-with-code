from pathlib import Path
import subprocess
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent

def main():
    base = Image.open(ROOT / 'first_frame.png').convert('RGB')
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    proc = subprocess.Popen([
        'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
        '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024',
        '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '12',
        '-preset', 'slow', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
        str(out / 'video.mp4')
    ], stdin=subprocess.PIPE)
    scale = 4
    for i in range(48):
        frame = base.copy()
        if i:
            # Smooth antialiasing is confined to the red pen stroke.
            overlay = Image.new('RGBA', (1024*scale, 1024*scale))
            draw = ImageDraw.Draw(overlay)
            progress = min(i / 45.0, 1.0)
            angles = np.linspace(-np.pi/2, -np.pi/2 + 2*np.pi*progress,
                                 max(2, int(progress*720)))
            points = [((504 + 89*np.cos(a))*scale,
                       (174 + 89*np.sin(a))*scale) for a in angles]
            draw.line(points, fill=(230, 25, 30, 255), width=5*scale, joint='curve')
            r=2.5*scale
            for x,y in (points[0], points[-1]):
                draw.ellipse((x-r,y-r,x+r,y+r), fill=(230,25,30,255))
            overlay = overlay.resize(base.size, Image.Resampling.LANCZOS)
            frame.paste(overlay, (0,0), overlay)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait():
        raise RuntimeError('ffmpeg failed')

if __name__ == '__main__':
    main()
