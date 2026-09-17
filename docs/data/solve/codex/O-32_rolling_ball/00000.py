from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import subprocess

ROOT = Path('/app')

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    ball = (original[:,:,0] == original[:,:,1]) & (original[:,:,2] == 0)
    background = original.copy()
    background[ball] = 255
    # Extract the exact, untextured yellow ball, including its olive outline.
    rgba = np.zeros((1024,1024,4), dtype=np.uint8)
    rgba[:,:,:3] = original
    rgba[:,:,3] = ball.astype(np.uint8)*255
    blue = (original[:,:,2] > 100) & (original[:,:,0] < 130) & (original[:,:,1] < 180)
    count, _, stats, centers = cv2.connectedComponentsWithStats(blue.astype(np.uint8))
    platforms = sorted([tuple(centers[i]) for i in range(1,count) if stats[i,4]>50])
    # The leftmost tile is partially occluded in the initial image.
    platforms[0] = (404.5,571.5)
    points = np.array(platforms, dtype=float)
    # Ball radius plus the height of the top face places it on each tile.
    points[:,1] -= 47.0
    frames=[]
    for frame in range(64):
        if frame == 0:
            frames.append(original.copy())
            continue
        if frame <= 9:
            t=frame/9.0
            s=t*t*(3-2*t)
            pos=np.array([368.,592.])*(1-s)+points[0]*s
            pos[1] -= 12*np.sin(np.pi*t)**2
        elif frame < 60:
            t=(frame-9)/50.0
            # Slow down gently as the ball approaches the last platform.
            progress=(1-(1-t)**1.25)*(len(points)-1)
            i=min(int(progress),len(points)-2)
            f=progress-i
            pos=points[i]*(1-f)+points[i+1]*f
            # Small rise between successive contacts, zero at every landing.
            pos[1] -= 2.8*np.sin(np.pi*f)**2
        else:
            pos=points[-1].copy()
        transform=np.float32([[1,0,pos[0]-368],[0,1,pos[1]-592]])
        moved=cv2.warpAffine(rgba,transform,(1024,1024),flags=cv2.INTER_LINEAR,borderValue=(0,0,0,0))
        alpha=moved[:,:,3:4].astype(float)/255
        result=np.rint(background*(1-alpha)+moved[:,:,:3]*alpha).astype(np.uint8)
        frames.append(result)
    out=ROOT/'output'
    out.mkdir(exist_ok=True)
    command=['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pixel_format','rgb24','-video_size','1024x1024','-framerate','16','-i','-','-an','-c:v','libx264','-crf','16','-pix_fmt','yuv420p','-movflags','+faststart',str(out/'video.mp4')]
    proc=subprocess.Popen(command,stdin=subprocess.PIPE)
    for frame in frames:
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait():
        raise RuntimeError('ffmpeg encoding failed')

if __name__=='__main__':
    main()
