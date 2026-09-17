from pathlib import Path
import subprocess
import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent

def main():
    base = Image.open(ROOT / 'first_frame.png').convert('RGB')
    arr = np.asarray(base)
    n, _, stats, centers = cv2.connectedComponentsWithStats((arr.min(axis=2) < 100).astype(np.uint8))
    dots = sorted([tuple(centers[i]) for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] > 100])
    cx, cy = dots[len(dots)//2]
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    command = ['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pixel_format','rgb24','-video_size','1024x1024','-framerate','16','-i','-','-an','-c:v','libx264','-preset','slow','-crf','0','-pix_fmt','yuv420p',str(out / 'video.mp4')]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE)
    scale = 4
    radius = 25
    for frame in range(22):
        image = base.copy()
        if frame:
            mask = Image.new('L', (1024*scale,1024*scale), 0)
            draw = ImageDraw.Draw(mask)
            box = tuple(int(v*scale) for v in (cx-radius,cy-radius,cx+radius,cy+radius))
            draw.arc(box, start=-90, end=-90+360*frame/21, fill=255, width=3*scale)
            mask = mask.resize(base.size, Image.Resampling.LANCZOS)
            image.paste((255,0,0), (0,0), mask)
        proc.stdin.write(np.asarray(image).tobytes())
    proc.stdin.close()
    if proc.wait():
        raise RuntimeError('ffmpeg failed')

if __name__ == '__main__':
    main()
