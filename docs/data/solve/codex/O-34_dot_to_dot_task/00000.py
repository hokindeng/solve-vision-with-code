from pathlib import Path
import subprocess
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path('/app')
OUT = ROOT / 'output/video.mp4'
POINTS = [(798,459), (450,509), (755,667), (258,292),
          (346,867), (875,230), (597,257), (551,844)]

def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    base = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    h, w = base.shape[:2]
    yy, xx = np.mgrid[:h,:w]
    protected = np.any(base != 255, axis=2)
    for x,y in POINTS:
        protected |= (xx-x)**2 + (yy-y)**2 <= 45.5**2
    encoder = subprocess.Popen([
        'ffmpeg','-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo',
        '-pix_fmt','rgb24','-s',f'{w}x{h}','-r','16','-i','-',
        '-an','-c:v','libx264','-preset','slow','-crf','16','-pix_fmt','yuv420p',
        '-movflags','+faststart',str(OUT)],stdin=subprocess.PIPE)
    scale=4
    for frame in range(115):
        mask=Image.new('L',(w*scale,h*scale),0)
        draw=ImageDraw.Draw(mask)
        progress=7*frame/112
        for i in range(7):
            t=min(1.,max(0.,progress-i))
            if t<=0: continue
            x,y=POINTS[i]
            ex,ey=POINTS[i+1]
            end=(x+(ex-x)*t,y+(ey-y)*t)
            draw.line((x*scale,y*scale,end[0]*scale,end[1]*scale),fill=255,width=4*scale)
        alpha=np.array(mask.resize((w,h),Image.Resampling.LANCZOS),dtype=np.float32)/255
        alpha[protected]=0
        rgb=np.rint(base*(1-alpha[:,:,None])+np.array([255,0,0])*alpha[:,:,None]).astype(np.uint8)
        encoder.stdin.write(rgb.tobytes())
    encoder.stdin.close()
    if encoder.wait(): raise RuntimeError('ffmpeg encoding failed')

if __name__=='__main__': main()
