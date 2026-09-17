from pathlib import Path
import subprocess
import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent

def main():
    base = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    gray = cv2.cvtColor(base, cv2.COLOR_RGB2GRAY)
    _, _, stats, centers = cv2.connectedComponentsWithStats((gray < 128).astype(np.uint8))
    hollow = []
    for (x, y, w, h, area), (cx, cy) in zip(stats[1:], centers[1:]):
        if w > 50 and h > 50 and area < w*h*0.2:
            hollow.append((float(cx), float(cy), max(w,h)/2 + 8))
    hollow.sort(key=lambda p: (p[1], p[0]))
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    proc = subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-crf','15','-preset','medium','-pix_fmt','yuv420p','-movflags','+faststart',str(out/'video.mp4')], stdin=subprocess.PIPE)
    scale = 4
    for frame in range(80):
        mask = Image.new('L',(1024*scale,1024*scale),0)
        draw = ImageDraw.Draw(mask)
        for i,(cx,cy,r) in enumerate(hollow):
            progress = min(1., max(0., (frame - (4 + i*12))/10))
            if progress == 0: continue
            box = tuple(int(round(v*scale)) for v in (cx-r,cy-r,cx+r,cy+r))
            draw.arc(box, -90, -90+360*progress, fill=255, width=4*scale)
        alpha = np.array(mask.resize((1024,1024),Image.Resampling.LANCZOS)).astype(np.float32)/255
        result = base.copy()
        active = alpha > 0
        a = alpha[active,None]
        result[active] = np.round(base[active]*(1-a)+np.array([235,20,30])*a).astype(np.uint8)
        proc.stdin.write(result.tobytes())
    proc.stdin.close()
    if proc.wait(): raise RuntimeError('ffmpeg failed')

if __name__ == '__main__':
    main()
