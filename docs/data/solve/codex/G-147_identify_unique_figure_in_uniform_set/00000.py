from pathlib import Path
import subprocess
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
base = Image.open(ROOT / 'first_frame.png').convert('RGB')
# The two circles match; the rectangle is the single different figure.
# Pause to inspect, trace its enclosing circle, then hold the solution.
fps, frames = 16, 60
scale = 4
cx, cy, radius = 495.5, 400, 94
box = (int(cx-radius-6), int(cy-radius-6), int(cx+radius+7), int(cy+radius+7))
w, h = box[2]-box[0], box[3]-box[1]
cmd = ['ffmpeg','-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo',
       '-pix_fmt','rgb24','-s','1024x1024','-r',str(fps),'-i','-',
       '-an','-c:v','libx264','-crf','0','-preset','medium','-pix_fmt','yuv420p',
       '-movflags','+faststart',str(OUT/'video.mp4')]
proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
for i in range(frames):
    frame = base.copy()
    if i >= 12:
        progress = min(1.0, (i-11)/40)
        overlay = Image.new('RGBA',(w*scale,h*scale),(0,0,0,0))
        draw = ImageDraw.Draw(overlay)
        bounds = tuple(v*scale for v in (cx-radius-box[0],cy-radius-box[1],cx+radius-box[0],cy+radius-box[1]))
        draw.arc(bounds,start=-90,end=-90+360*progress,fill=(230,30,35,255),width=5*scale)
        overlay = overlay.resize((w,h),Image.Resampling.LANCZOS)
        frame.paste(overlay,(box[0],box[1]),overlay)
    proc.stdin.write(np.asarray(frame).tobytes())
proc.stdin.close()
if proc.wait() != 0:
    raise RuntimeError('ffmpeg failed')
