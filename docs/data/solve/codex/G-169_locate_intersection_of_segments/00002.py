from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
import subprocess

ROOT = Path('/app')

def main():
    original = Image.open(ROOT / 'first_frame.png').convert('RGB')
    a = np.array(original)
    mask = (a[:,:,2] > 140) & (a[:,:,0] < 160) & (a[:,:,1] < 150)
    y, x = np.where(mask)
    # Fit each segment away from the crossing, then solve for their intersection.
    horizontal = (x < 270) | (x > 505)
    horizontal &= (y > 730) & (y < 775)
    diagonal = (y < 715) | (y > 800)
    mh, bh = np.polyfit(x[horizontal], y[horizontal], 1)
    md, bd = np.polyfit(x[diagonal], y[diagonal], 1)
    cx = (bd - bh) / (mh - md)
    cy = mh * cx + bh
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    command = ['ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
               '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16', '-i', '-',
               '-an', '-c:v', 'libx264', '-preset', 'slow', '-crf', '0',
               '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out / 'video.mp4')]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    scale = 4
    radius = 29
    for frame in range(30):
        result = original.copy()
        progress = np.clip((frame - 2) / 25, 0, 1)
        if progress > 0:
            layer = Image.new('RGBA', (4096,4096))
            draw = ImageDraw.Draw(layer)
            angles = np.linspace(-np.pi / 2, -np.pi / 2 + 2*np.pi*progress, max(2,int(240*progress)))
            points = [((cx + radius*np.cos(t))*scale,(cy + radius*np.sin(t))*scale) for t in angles]
            draw.line(points, fill=(230,35,40,255), width=4*scale, joint='curve')
            for px,py in (points[0],points[-1]):
                draw.ellipse((px-2*scale,py-2*scale,px+2*scale,py+2*scale), fill=(230,35,40,255))
            layer = layer.resize(original.size, Image.Resampling.LANCZOS)
            result.paste(layer, (0,0), layer)
        process.stdin.write(np.array(result).tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError('ffmpeg failed')
    print(f'Intersection: ({cx:.2f}, {cy:.2f})')

if __name__ == '__main__':
    main()
