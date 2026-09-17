from pathlib import Path
import cv2
import numpy as np
import subprocess

ROOT = Path('/app')

def ease(t):
    t = np.clip(t, 0., 1.)
    return t*t*(3-2*t)

def main():
    original = cv2.imread(str(ROOT/'first_frame.png'))
    h,w = original.shape[:2]
    foreground = (np.any(original < 250,axis=2) & (np.indices((h,w))[1] < 450)).astype(np.uint8)
    n,labels,stats,centers = cv2.connectedComponentsWithStats(foreground)
    background = original.copy()
    background[foreground.astype(bool)] = 255
    objects=[]
    # Each target has the same vertical center as its source.
    for k in range(1,n):
        if stats[k,cv2.CC_STAT_AREA] < 100: continue
        mask=(labels==k)
        y,x=np.where(mask)
        rect=cv2.minAreaRect(np.column_stack((x,y)).astype(np.float32))
        cx,cy=rect[0]
        angle=rect[2]
        while angle < -45: angle+=90
        if cx < 150:
            target_x,target_angle=602.16,18.825
        elif cy > 550:
            target_x,target_angle=805.11,23.9625
        else:
            target_x,target_angle=805.06,-9.0193
        sprite=np.full_like(original,255)
        sprite[mask]=original[mask]
        objects.append((sprite,(cx,cy),target_x,target_angle-angle))
    out=ROOT/'output'
    out.mkdir(exist_ok=True)
    proc=subprocess.Popen(['ffmpeg','-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','bgr24','-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-crf','17','-pix_fmt','yuv420p','-movflags','+faststart',str(out/'video.mp4')],stdin=subprocess.PIPE,stderr=subprocess.DEVNULL)
    targets=(np.indices((h,w))[1]>500)&np.any(original<255,axis=2)
    for frame in range(48):
        if frame==0:
            result=original.copy()
        else:
            rotation=ease(frame/19)
            translation=ease((frame-19)/28)
            result=background.copy()
            for sprite,center,tx,delta in objects:
                matrix=cv2.getRotationMatrix2D(center,-delta*rotation,1.)
                matrix[0,2]+=(tx-center[0])*translation
                warped=cv2.warpAffine(sprite,matrix,(w,h),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_CONSTANT,borderValue=(255,255,255))
                result=np.minimum(result,warped)
            # Preserve every stationary dash pixel, including its antialiasing.
            result[targets]=original[targets]
        proc.stdin.write(result.tobytes())
    proc.stdin.close()
    if proc.wait()!=0: raise RuntimeError('ffmpeg failed')

if __name__=='__main__':
    main()
