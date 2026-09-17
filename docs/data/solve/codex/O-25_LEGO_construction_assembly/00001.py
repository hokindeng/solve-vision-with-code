from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import cv2
import subprocess

ROOT=Path('/app')
def main():
    original=Image.open(ROOT/'first_frame.png').convert('RGB')
    a=np.array(original)
    # The brick is translated in the same isometric orientation as the target.
    top=[(30,402),(203,302),(290,352),(116,451)]
    left=[(30,402),(116,451),(116,510),(30,462)]
    right=[(116,451),(290,352),(290,412),(116,510)]
    outline=[(30,402),(203,302),(290,352),(290,412),(116,510),(30,462)]
    mask=Image.new('L',original.size,0)
    ImageDraw.Draw(mask).polygon(outline,fill=255)
    m=np.array(mask)>0
    clean=Image.new('RGB',original.size,(245,245,240))
    d=ImageDraw.Draw(clean)
    d.rectangle((70,331,242,501),fill=(255,255,255),outline=(178,178,178),width=2)
    base=a.copy()
    removal=cv2.dilate(m.astype(np.uint8),np.ones((3,3),np.uint8))>0
    base[removal]=np.array(clean)[removal]
    # Retain the instruction arrow where it originally crossed the callout brick.
    arrow=m & (a[:,:,0]>220) & ((a[:,:,1]<100) | ((a[:,:,1]>220)&(a[:,:,2]>220)))
    base[arrow]=a[arrow]
    painted=Image.new('RGB',original.size)
    d=ImageDraw.Draw(painted)
    d.polygon(top,fill=(0,102,229),outline=(50,50,50))
    d.polygon(left,fill=(0,59,133),outline=(50,50,50))
    d.polygon(right,fill=(0,72,162),outline=(50,50,50))
    p=np.array(painted)
    # Copy the original studs, outlines, and face colors without the overlaid arrow.
    blue=(a[:,:,2]>80)&(a[:,:,0]<70)&(a[:,:,2]>a[:,:,1])
    dark=(a.max(axis=2)<100)
    p[m&(blue|dark)]=a[m&(blue|dark)]
    sprite=Image.fromarray(p).convert('RGBA')
    sprite.putalpha(mask)
    sprite=sprite.crop((29,301,292,512))
    frames=[]
    for i in range(46):
        if i==0:
            frame=a.copy()
        else:
            t=min(i/43,1.0)
            s=t*t*(3-2*t)
            dx=262*s
            dy=446*s-65*np.sin(np.pi*s)
            # Finish with a short, straight seating movement.
            background=base.copy()
            if s>0.99:
                target=np.zeros_like(m,dtype=np.uint8)
                target[446:,262:]=np.array(mask)[:-446,:-262]
                target=cv2.dilate(target,np.ones((5,5),np.uint8))>0
                red=(a[:,:,0]>240)&(a[:,:,1]<50)&(a[:,:,2]<50)
                erase=target&red
                background[erase]=(245,245,240)
            canvas=Image.fromarray(background).convert('RGBA')
            canvas.alpha_composite(sprite,(round(29+dx),round(301+dy)))
            frame=np.array(canvas.convert('RGB'))
        frames.append(frame)
    out=ROOT/'output'
    out.mkdir(exist_ok=True)
    proc=subprocess.Popen(['ffmpeg','-y','-f','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-crf','15','-pix_fmt','yuv420p','-movflags','+faststart',str(out/'video.mp4')],stdin=subprocess.PIPE,stderr=subprocess.DEVNULL)
    for frame in frames:
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait(): raise RuntimeError('ffmpeg failed')
    Image.fromarray(frames[-1]).save(out/'last_frame.png')
if __name__=='__main__': main()
