from pathlib import Path
import math
import numpy as np
from PIL import Image, ImageDraw
import subprocess

ROOT = Path('/app')
OUT = ROOT / 'output'
# Vertices in clockwise order, including the narrow inward notch.
VERTICES = [(142,456),(906,456),(789,659),(770,898),
            (673,919),(645,859),(638,927),(199,714)]

def main():
    OUT.mkdir(exist_ok=True)
    source = Image.open(ROOT / 'first_frame.png').convert('RGB')
    lengths = [math.dist(VERTICES[i], VERTICES[(i+1)%8]) for i in range(8)]
    longest = int(np.argmax(lengths))
    a, b = VERTICES[longest], VERTICES[(longest+1)%8]
    midpoint = ((a[0]+b[0])/2, (a[1]+b[1])/2)
    print('Edge lengths (pixels):', [round(x,2) for x in lengths])
    print('Longest edge midpoint:', midpoint)
    # Draw overlays at 4x resolution and composite only their nonzero pixels.
    def overlay(frame):
        layer = Image.new('RGBA',(4096,4096))
        d = ImageDraw.Draw(layer)
        if 1 <= frame <= 16:
            # Inspect all eight edges, one at a time. The second beat traces
            # the full edge, making each length directly visible.
            edge = (frame-1)//2
            p, q = VERTICES[edge], VERTICES[(edge+1)%8]
            fraction = 0.5 if frame%2 else 1.0
            end = (p[0]+fraction*(q[0]-p[0]),p[1]+fraction*(q[1]-p[1]))
            d.line([(p[0]*4,p[1]*4),(end[0]*4,end[1]*4)], fill=(50,110,180,255),width=8)
        elif 17 <= frame <= 18:
            d.line([(a[0]*4,a[1]*4),(b[0]*4,b[1]*4)],fill=(50,110,180,255),width=8)
        elif frame >= 19:
            x,y = midpoint
            radius = 10
            box = tuple(v*4 for v in (x-radius,y-radius,x+radius,y+radius))
            angle = 360*min(1,(frame-18)/5)
            d.arc(box,start=-90,end=-90+angle,fill=(230,25,35,255),width=10)
        layer = layer.resize(source.size,Image.Resampling.LANCZOS)
        return Image.alpha_composite(source.convert('RGBA'),layer).convert('RGB')
    command = ['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','rgb24',
               '-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264',
               '-crf','10','-preset','slow','-pix_fmt','yuv420p',
               '-movflags','+faststart',str(OUT/'video.mp4')]
    process = subprocess.Popen(command,stdin=subprocess.PIPE)
    for frame in range(25):
        image = source if frame == 0 else overlay(frame)
        process.stdin.write(image.tobytes())
    process.stdin.close()
    if process.wait():
        raise RuntimeError('ffmpeg encoding failed')

if __name__ == '__main__':
    main()
