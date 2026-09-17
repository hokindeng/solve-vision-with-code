from pathlib import Path
import math
import subprocess
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path('/app')
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
h, w = original.shape[:2]
# Isolate the angle label and its black arc, preserving the normal and ray.
annotation = np.zeros((h,w), dtype=bool)
annotation[468:495, 555:650] = np.any(original[468:495,555:650] < 255, axis=2)
arc = original[471:492,477:514]
annotation[471:492,477:514] |= (arc.max(axis=2) < 100)
angle = math.asin(math.sin(math.radians(56.3))/1.857)
start = np.array([512.0, 514.0])
end = np.array([512.0 + (1023.0-514.0)*math.tan(angle),1023.0])

command = ['ffmpeg','-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24',
           '-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264',
           '-preset','slow','-crf','0','-pix_fmt','yuv420p','-movflags','+faststart',
           str(OUT/'video.mp4')]
proc = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
for frame in range(70):
    canvas = original.copy()
    fade = min(1.0,frame/12.0)
    canvas[annotation] = np.rint(original[annotation]*(1-fade)+255*fade).astype(np.uint8)
    progress = np.clip((frame-10)/59.0,0,1)
    if progress > 0:
        tip = start+(end-start)*progress
        # Supersample only the new ray; all pixels outside its coverage stay original.
        scale = 4
        mask = Image.new('L',(w*scale,h*scale),0)
        draw = ImageDraw.Draw(mask)
        draw.line([tuple(start*scale),tuple(tip*scale)],fill=255,width=8)
        direction = np.array([math.sin(angle),math.cos(angle)])
        side = np.array([direction[1],-direction[0]])
        head = min(15.0,np.linalg.norm(tip-start)*0.45)
        for sign in [-1,1]:
            tail = tip-direction*head+sign*side*head*0.35
            draw.line([tuple(tail*scale),tuple(tip*scale)],fill=255,width=7)
        alpha = np.array(mask.resize((w,h),Image.Resampling.LANCZOS))/255.0
        alpha[:514,:] = 0
        active = alpha > 0
        a = alpha[active,None]
        canvas[active] = np.rint(canvas[active]*(1-a)+np.array([255,0,0])*a).astype(np.uint8)
    proc.stdin.write(canvas.tobytes())
proc.stdin.close()
err = proc.stderr.read()
if proc.wait():
    raise RuntimeError(err.decode())
print(f'Refraction angle: {math.degrees(angle):.4f} degrees; boundary x: {end[0]:.3f}')
