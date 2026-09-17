from PIL import Image, ImageDraw
import numpy as np
import cv2
import math
from pathlib import Path

ROOT = Path('/app')

def main():
    base = Image.open(ROOT / 'first_frame.png').convert('RGB')
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    # The six values are 15, 83, 87, 44, 31, 83; only 87 is marked.
    frames = 80
    scale = 4
    writer = cv2.VideoWriter(str(out / 'video.mp4'), cv2.VideoWriter_fourcc(*'avc1'), 16, (1024, 1024))
    if not writer.isOpened():
        import subprocess
        proc = subprocess.Popen(['ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p', str(out / 'video.mp4')], stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    else:
        proc = None
    for i in range(frames):
        frame = base.copy()
        # Initial pause allows comparison; then trace one continuous ellipse.
        progress = min(1.0, max(0.0, (i - 12) / 60))
        if progress > 0:
            overlay = Image.new('RGBA', (1024*scale, 1024*scale))
            draw = ImageDraw.Draw(overlay)
            angles = np.linspace(-math.pi/2, -math.pi/2 + 2*math.pi*progress, max(2, int(600*progress)))
            points = [((532 + 101*math.cos(t))*scale, (355 + 68*math.sin(t))*scale) for t in angles]
            draw.line(points, fill=(235, 25, 35, 255), width=5*scale, joint='curve')
            for x,y in (points[0], points[-1]):
                r=2.5*scale
                draw.ellipse((x-r,y-r,x+r,y+r), fill=(235,25,35,255))
            overlay = overlay.resize(base.size, Image.Resampling.LANCZOS)
            frame.paste(overlay, (0,0), overlay)
        array = np.asarray(frame)
        if proc:
            proc.stdin.write(array.tobytes())
        else:
            writer.write(cv2.cvtColor(array, cv2.COLOR_RGB2BGR))
    if proc:
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError('ffmpeg failed')
    else:
        writer.release()

if __name__ == '__main__':
    main()
