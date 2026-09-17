from pathlib import Path
import math
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import subprocess

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
base = Image.open(ROOT / 'first_frame.png').convert('RGB')
a = np.asarray(base)
_, _, stats, _ = cv2.connectedComponentsWithStats(np.uint8(np.any(a < 240, axis=2)))
rects = [tuple(map(int, r[:4])) for r in stats[1:] if r[4] > 100]
closest = min(rects, key=lambda r: abs(r[2]/r[3]-1))
font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
proc = subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pixel_format','rgb24','-video_size','1024x1024','-framerate','16','-i','-','-an','-c:v','libx264','-crf','0','-pix_fmt','yuv420p',str(OUT/'video.mp4')], stdin=subprocess.PIPE)
for frame in range(48):
    im = base.copy()
    draw = ImageDraw.Draw(im)
    # Compare each ratio with the square's ideal ratio of one.
    if 3 <= frame < 28:
        for index, (x,y,w,h) in enumerate(rects):
            if frame >= 3 + index*6:
                label = f'{w}:{h} = {w/h:.3f}   |ratio - 1| = {abs(w/h-1):.3f}'
                draw.text((x-10,y+h+14),label,font=font,fill=(35,35,35))
    # Remove the working notes, and draw exactly one circular selection.
    if frame >= 28:
        x,y,w,h = closest
        cx,cy = x+(w-1)/2,y+(h-1)/2
        radius = math.hypot(w,h)/2 + 13
        progress = min(1,(frame-27)/17)
        overlay = Image.new('RGBA',(4096,4096),(0,0,0,0))
        pen = ImageDraw.Draw(overlay)
        box = tuple(round(v*4) for v in (cx-radius,cy-radius,cx+radius,cy+radius))
        pen.arc(box,-90,-90+360*progress,fill=(235,30,35,255),width=16)
        overlay = overlay.resize(base.size,Image.Resampling.LANCZOS)
        im.paste(overlay,(0,0),overlay)
    proc.stdin.write(np.asarray(im).tobytes())
proc.stdin.close()
if proc.wait() != 0:
    raise RuntimeError('Video encoding failed')
