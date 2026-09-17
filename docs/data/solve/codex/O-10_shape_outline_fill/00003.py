from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import subprocess

ROOT=Path('/app')
def main():
    original=Image.open(ROOT/'first_frame.png').convert('RGB')
    base=np.array(original)
    blue=tuple(base[200,90])
    # The example exposes the outlines of each constituent primitive.
    # The arrow consists of a rectangular shaft and a triangular head.
    scale=4
    def mask(outline=False):
        im=Image.new('L',(1024*scale,1024*scale))
        d=ImageDraw.Draw(im)
        rect=[(784,728),(864,728),(864,808),(784,808)]
        tri=[(864,688),(944,768),(864,848)]
        for pts in (rect,tri):
            pts=[(x*scale,y*scale) for x,y in pts]
            if outline:
                d.line(pts+[pts[0]],fill=255,width=8*scale,joint='curve')
            else:
                d.polygon(pts,fill=255)
        return np.asarray(im.resize((1024,1024),Image.Resampling.LANCZOS))/255.
    fill=mask(); outline=mask(True)
    question=np.zeros((1024,1024),dtype=float)
    question[738:797,838:891]=1
    out=ROOT/'output'; out.mkdir(exist_ok=True)
    proc=subprocess.Popen(['ffmpeg','-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-crf','15','-pix_fmt','yuv420p',str(out/'video.mp4')],stdin=subprocess.PIPE,stderr=subprocess.DEVNULL)
    def smooth(t):
        t=np.clip(t,0,1); return t*t*(3-2*t)
    for i in range(60):
        a=base.astype(float)
        fade=smooth(i/14)
        q=question*fade
        a=a*(1-q[:,:,None])+255*q[:,:,None]
        appear=smooth((i-9)/18)
        hollow=smooth((i-25)/34)
        opacity=appear*np.maximum(fill*(1-hollow),outline*hollow)
        a=a*(1-opacity[:,:,None])+np.array(blue)*opacity[:,:,None]
        frame=np.clip(np.rint(a),0,255).astype('uint8')
        if i==0: assert np.array_equal(frame,base)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait(): raise RuntimeError('ffmpeg failed')
if __name__=='__main__': main()
