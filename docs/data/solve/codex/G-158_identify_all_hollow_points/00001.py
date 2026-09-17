from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
import subprocess

ROOT = Path('/app')

def main():
    original = Image.open(ROOT / 'first_frame.png').convert('RGB')
    gray = np.array(original.convert('L'))
    _, _, stats, centers = cv2.connectedComponentsWithStats((gray < 128).astype(np.uint8))
    hollow = []
    for (x,y,w,h,area), (cx,cy) in zip(stats[1:], centers[1:]):
        if w > 20 and h > 20 and area / (w*h) < 0.2:
            hollow.append((float(cx), float(cy), max(w,h)/2 + 9))
    hollow.sort(key=lambda p:(p[1],p[0]))
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    proc = subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p',str(out/'video.mp4')],stdin=subprocess.PIPE)
    scale=4
    for frame in range(80):
        # Begin from the exact source on every frame. Only ring pixels are overlaid.
        overlay=Image.new('RGBA',(1024*scale,1024*scale),(0,0,0,0))
        draw=ImageDraw.Draw(overlay)
        for idx,(cx,cy,r) in enumerate(hollow):
            start=6+idx*23
            progress=min(1.,max(0.,(frame-start)/21))
            if progress>0:
                box=tuple(v*scale for v in (cx-r,cy-r,cx+r,cy+r))
                draw.arc(box,-90,-90+360*progress,fill=(235,25,35,255),width=4*scale)
        overlay=overlay.resize(original.size,Image.Resampling.LANCZOS)
        result=Image.alpha_composite(original.convert('RGBA'),overlay).convert('RGB')
        proc.stdin.write(np.asarray(result).tobytes())
    proc.stdin.close()
    if proc.wait()!=0:
        raise RuntimeError('ffmpeg failed')

if __name__=='__main__':
    main()
