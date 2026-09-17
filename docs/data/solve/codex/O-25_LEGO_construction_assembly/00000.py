from PIL import Image, ImageDraw
import numpy as np
import cv2
import os

ROOT=os.path.dirname(os.path.abspath(__file__))
def main():
    original=Image.open(os.path.join(ROOT,'first_frame.png')).convert('RGB')
    a=np.array(original)
    # Isolate the illustrated brick, including its fine boundary strokes.
    mask=Image.new('L',original.size,0)
    ImageDraw.Draw(mask).polygon([(160,327),(247,377),(247,437),(160,486),(73,437),(73,377)],fill=255)
    m=np.array(mask)>0
    clean=Image.new('RGB',original.size,(245,245,240))
    d=ImageDraw.Draw(clean)
    d.rectangle((70,331,240,501), fill=(255,255,255),outline=(180,180,180),width=2)
    base=a.copy()
    base[m]=np.array(clean)[m]
    # The arrow is a fixed annotation and remains where it was drawn.
    annotation=m & (np.indices(m.shape)[0]>=410) & ((a[:,:,0]==255)&((a[:,:,1]==0)|(a[:,:,1]==255)))
    base[annotation]=a[annotation]
    sprite=a.copy()
    # Repair the small part of the brick covered by the annotation.
    for y,x in zip(*np.where(annotation)):
        edge=427-abs(x-160)*50/87
        sprite[y,x]=(241,31,10) if y<=edge else ((140,18,6) if x<160 else (170,22,7))
    rgba=np.dstack((sprite,np.array(mask)))
    brick=Image.fromarray(rgba).crop((72,326,249,488))
    os.makedirs(os.path.join(ROOT,'output'),exist_ok=True)
    import subprocess
    writer=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pixel_format','rgb24','-video_size','1024x1024','-framerate','16','-i','-','-c:v','libx264','-crf','0','-preset','slow','-pix_fmt','yuv420p','-movflags','+faststart',os.path.join(ROOT,'output','video.mp4')],stdin=subprocess.PIPE)
    for i in range(46):
        if i==0:
            frame=original
        else:
            t=min(i/43,1.)
            s=t*t*(3-2*t)
            # Follow a gentle lifted arc, slowing into the final clutch position.
            dx=435*s
            dy=496*s-130*np.sin(np.pi*s)
            frame=Image.fromarray(base)
            frame.paste(brick,(round(72+dx),round(326+dy)),brick)
        writer.stdin.write(np.asarray(frame).tobytes())
    writer.stdin.close()
    if writer.wait()!=0: raise RuntimeError('Video encoding failed')
if __name__=='__main__': main()
