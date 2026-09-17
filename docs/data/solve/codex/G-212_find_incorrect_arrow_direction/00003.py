from pathlib import Path
import math
import numpy as np
from PIL import Image, ImageDraw
import cv2

ROOT = Path('/app')

def main():
    original = Image.open(ROOT / 'first_frame.png').convert('RGB')
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    temp = out / 'drawing.avi'
    writer = cv2.VideoWriter(str(temp), cv2.VideoWriter_fourcc(*'FFV1'), 16, original.size)
    scale = 4
    for i in range(48):
        frame = original.copy()
        if i:
            progress = min(i / 44, 1.0)
            mask = Image.new('L', (1024 * scale, 1024 * scale), 0)
            draw = ImageDraw.Draw(mask)
            points = []
            for angle in np.linspace(-math.pi / 2, -math.pi / 2 + 2 * math.pi * progress, max(2, int(700 * progress))):
                points.append(((620 + 111 * math.cos(angle)) * scale, (812 + 70 * math.sin(angle)) * scale))
            draw.line(points, fill=255, width=5 * scale, joint='curve')
            r = 2.5 * scale
            for x,y in (points[0], points[-1]):
                draw.ellipse((x-r,y-r,x+r,y+r),fill=255)
            mask = mask.resize(original.size, Image.Resampling.LANCZOS)
            frame.paste((230, 20, 25), (0, 0), mask)
        writer.write(cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR))
    writer.release()
    import subprocess
    subprocess.run(['ffmpeg','-y','-i',str(temp),'-c:v','libx264','-crf','15','-pix_fmt','yuv420p','-movflags','+faststart',str(out/'video.mp4')], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    temp.unlink()

if __name__ == '__main__':
    main()
