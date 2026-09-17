from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import cv2
import subprocess

ROOT = Path('/app')

def main():
    base = Image.open(ROOT/'first_frame.png').convert('RGB')
    a = np.asarray(base)
    orange = (a[:,:,0]>220)&(a[:,:,1]>100)&(a[:,:,1]<190)&(a[:,:,2]<70)
    purple = (a[:,:,0]>100)&(a[:,:,0]<190)&(a[:,:,1]<110)&(a[:,:,2]>120)
    def fit(mask):
        y,x=np.where(mask)
        vx,vy,x0,y0=cv2.fitLine(np.column_stack((x,y)).astype(np.float32),cv2.DIST_L2,0,0.01,0.01).flatten()
        return np.array([x0,y0]),np.array([vx,vy])
    p,u=fit(orange); q,v=fit(purple)
    t=np.linalg.solve(np.column_stack((u,-v)),q-p)[0]
    cx,cy=p+t*u
    out=ROOT/'output'
    out.mkdir(exist_ok=True)
    enc=subprocess.Popen(['ffmpeg','-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(out/'video.mp4')],stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    scale=4
    for i in range(30):
        frame=base.copy()
        if i>=3:
            progress=min(1,(i-2)/25)
            overlay=Image.new('RGBA',(1024*scale,1024*scale),(0,0,0,0))
            draw=ImageDraw.Draw(overlay)
            angles=np.linspace(-np.pi/2,-np.pi/2+2*np.pi*progress,max(2,int(360*progress)))
            pts=[((float(cx)+29*np.cos(t))*scale,(float(cy)+29*np.sin(t))*scale) for t in angles]
            draw.line(pts,fill=(230,28,35,255),width=4*scale,joint='curve')
            for x,y in (pts[0],pts[-1]):
                draw.ellipse((x-2*scale,y-2*scale,x+2*scale,y+2*scale),fill=(230,28,35,255))
            overlay=overlay.resize(base.size,Image.Resampling.LANCZOS)
            frame.paste(overlay,mask=overlay.getchannel('A'))
        enc.stdin.write(np.asarray(frame).tobytes())
    enc.stdin.close()
    error=enc.stderr.read()
    if enc.wait():
        raise RuntimeError(error.decode())
    print(f'Intersection: ({cx:.2f}, {cy:.2f})')

if __name__=='__main__':
    main()
