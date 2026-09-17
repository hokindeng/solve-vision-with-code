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
        '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'slow',
        '-crf', '0', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
        str(out / 'video.mp4')
    ], stdin=subprocess.PIPE)
    # Draw only the circle mask; retain all source pixels outside its stroke.
    scale = 4
    for frame in range(48):
        result = base.copy()
        if frame:
            progress = min(frame / 45.0, 1.0)
            angles = np.linspace(-np.pi/2, -np.pi/2 + 2*np.pi*progress,
                                 max(2, int(720*progress)))
            points = [((665 + 91*np.cos(a))*scale,
                       (413 + 91*np.sin(a))*scale) for a in angles]
            mask = Image.new('L', (1024*scale, 1024*scale), 0)
            draw = ImageDraw.Draw(mask)
            draw.line(points, fill=255, width=5*scale, joint='curve')
            for x,y in (points[0], points[-1]):
                r = 2.5*scale
                draw.ellipse((x-r,y-r,x+r,y+r), fill=255)
            mask = mask.resize(base.size, Image.Resampling.LANCZOS)
            result.paste((230, 20, 25), (0,0), mask)
        proc.stdin.write(np.asarray(result).tobytes())
    proc.stdin.close()
    if proc.wait():
        raise RuntimeError('ffmpeg failed')

if __name__ == '__main__':
    main()
