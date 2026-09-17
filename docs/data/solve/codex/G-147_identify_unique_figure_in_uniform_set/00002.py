from pathlib import Path
import math
import numpy as np
from PIL import Image, ImageDraw
import cv2
import subprocess

ROOT = Path('/app')
OUT = ROOT / 'output'

def main():
    OUT.mkdir(exist_ok=True)
    base = Image.open(ROOT / 'first_frame.png').convert('RGB')
    pixels = np.asarray(base)
    mask = ((pixels[:,:,1] > 150) & (pixels[:,:,0] < 180) & (pixels[:,:,2] < 100)).astype('uint8')
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    shapes = []
    for i in range(1, count):
        x,y,w,h,area = stats[i]
        if area > 100:
            shapes.append((area/(w*h), x,y,w,h))
    # The two rectangles fill their bounding boxes; the triangle fills half.
    _,x,y,w,h = min(shapes)
    cx = x + (w-1)/2
    cy = y + (h-1)*2/3
    radius = max(math.hypot(px-cx,py-cy) for px,py in
                 [(cx,y),(x,y+h-1),(x+w-1,y+h-1)]) + 10
    command = ['ffmpeg','-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo',
               '-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-',
               '-an','-c:v','libx264','-crf','15','-pix_fmt','yuv420p',
               '-movflags','+faststart',str(OUT/'video.mp4')]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    for frame in range(60):
        # Inspect the original arrangement, trace the answer, then hold it.
        progress = min(1.0, max(0.0,(frame-12)/39))
        result = base.copy()
        if progress > 0:
            scale = 4
            layer = Image.new('RGBA',(1024*scale,1024*scale))
            draw = ImageDraw.Draw(layer)
            angles = np.linspace(-math.pi/2,-math.pi/2+2*math.pi*progress,
                                 max(2,int(500*progress)))
            points = [((cx+radius*math.cos(a))*scale,
                       (cy+radius*math.sin(a))*scale) for a in angles]
            draw.line(points,fill=(235,25,35,255),width=5*scale,joint='curve')
            for px,py in (points[0],points[-1]):
                r=2.5*scale
                draw.ellipse((px-r,py-r,px+r,py+r),fill=(235,25,35,255))
            layer = layer.resize(base.size,Image.Resampling.LANCZOS)
            result.paste(layer,(0,0),layer)
        process.stdin.write(np.asarray(result).tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError('Video encoding failed')

if __name__ == '__main__':
    main()
