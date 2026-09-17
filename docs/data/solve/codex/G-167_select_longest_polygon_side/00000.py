from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
import subprocess

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
base = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
# Vertices follow the eight original boundary segments in clockwise order.
vertices = np.array([(134,503),(857,503),(698,597),(549,614),
                     (502,915),(447,751),(336,809),(285,692)], dtype=float)
lengths = np.linalg.norm(np.roll(vertices,-1,axis=0)-vertices,axis=1)
longest = int(np.argmax(lengths))
midpoint = (vertices[longest]+vertices[(longest+1)%8])/2
print('Edge lengths (pixels):', ', '.join(f'{x:.1f}' for x in lengths))
print('Longest edge midpoint:', midpoint.tolist())
# Comparison traces recolor only existing dark boundary pixels, leaving all
# background and polygon fill pixels untouched. The final mark is a red ring.
dark = np.max(base,axis=2) < 180
frames = [base.copy()]
for edge in range(8):
    a = vertices[edge]
    b = vertices[(edge+1)%8]
    for progress in (0.5,1.0):
        frame = base.copy()
        mask = np.zeros(base.shape[:2],np.uint8)
        end = a+(b-a)*progress
        cv2.line(mask,tuple(a.astype(int)),tuple(end.astype(int)),255,3)
        frame[(mask>0)&dark] = (40,105,225)
        frames.append(frame)
# Eight frames smoothly complete the small circle, with a final hold.
scale=4
for step in range(8):
    overlay = Image.new('RGBA',(1024*scale,1024*scale))
    draw = ImageDraw.Draw(overlay)
    x,y = midpoint
    radius=9
    bounds=tuple(round(v*scale) for v in (x-radius,y-radius,x+radius,y+radius))
    extent=360*min((step+1)/6,1)
    draw.arc(bounds,-90,-90+extent,fill=(230,30,40,255),width=3*scale)
    overlay=overlay.resize((1024,1024),Image.Resampling.LANCZOS)
    result=Image.alpha_composite(Image.fromarray(base).convert('RGBA'),overlay).convert('RGB')
    frames.append(np.array(result))
assert len(frames)==25 and np.array_equal(frames[0],base)
command=['ffmpeg','-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo',
         '-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-', '-an',
         '-c:v','libx264','-preset','slow','-crf','0','-pix_fmt','yuv420p',
         '-movflags','+faststart',str(OUT/'video.mp4')]
with subprocess.Popen(command,stdin=subprocess.PIPE) as encoder:
    for frame in frames:
        encoder.stdin.write(frame.tobytes())
    encoder.stdin.close()
    if encoder.wait():
        raise RuntimeError('Video encoding failed')

if __name__=='__main__':
    print(OUT/'video.mp4')
