from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import cv2

ROOT=Path('/app')

def main():
    original=Image.open(ROOT/'first_frame.png').convert('RGB')
    a=np.array(original)
    # Isometric faces of the loose 2 x 4 brick.
    top=[(30,401),(203,302),(290,352),(116,451)]
    left=[(30,401),(116,451),(116,511),(30,461)]
    front=[(116,451),(290,352),(290,412),(116,511)]
    outline=[(30,401),(203,302),(290,352),(290,412),(116,511),(30,461)]
    mask=Image.new('L',original.size,0)
    ImageDraw.Draw(mask).polygon(outline,fill=255)
    m=np.array(mask)>0
    clean=Image.new('RGB',original.size)
    d=ImageDraw.Draw(clean)
    for face,color in [(left,(177,96,16)),(front,(215,117,20)),(top,(255,165,28))]:
        d.polygon(face,fill=color,outline=(50,50,50))
    # Retain the original stud drawing and all unoccluded brick pixels.
    brick=a.copy()
    yy,xx=np.indices(m.shape)
    arrow=((a[:,:,0]>240)&(a[:,:,1]<100)&(a[:,:,2]<100)) | np.all(a==255,axis=2)
    repair=m & arrow & (xx>150)&(yy>410)
    brick[repair]=np.array(clean)[repair]
    sprite=Image.fromarray(brick).convert('RGBA')
    sprite.putalpha(mask)
    sprite=sprite.crop((29,301,292,513))

    # Reconstruct only the area formerly hidden by the loose brick.
    under=Image.new('RGB',original.size,(245,245,240))
    du=ImageDraw.Draw(under)
    du.rectangle((70,331,240,501),fill='white',outline=(178,178,178),width=2)
    base=original.copy()
    removal=mask.filter(ImageFilter.MaxFilter(5))
    base.paste(under,(0,0),removal)
    b=np.array(base)
    # Instruction arrow is a fixed diagram element, including its white keyline.
    keep=(np.array(removal)>0)&arrow&(xx>150)&(yy>410)
    b[keep]=a[keep]
    base=Image.fromarray(b)

    output=ROOT/'output'
    output.mkdir(exist_ok=True)
    import subprocess
    writer=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-crf','0','-pix_fmt','yuv420p','-movflags','+faststart',str(output/'video.mp4')],stdin=subprocess.PIPE)
    for i in range(46):
        if i==0:
            frame=original
        else:
            # Smooth acceleration and deceleration, with a slight lifted approach.
            t=min(1.0,i/43.0)
            s=t*t*(3-2*t)
            dx=522*s
            dy=488*s-65*np.sin(np.pi*s)
            frame=base.copy()
            frame.paste(sprite,(round(29+dx),round(301+dy)),sprite)
        writer.stdin.write(np.asarray(frame).tobytes())
        if i==45: frame.save(output/'last_frame.png')
    writer.stdin.close()
    if writer.wait()!=0:
        raise RuntimeError("Video encoding failed")

if __name__=='__main__':
    main()
