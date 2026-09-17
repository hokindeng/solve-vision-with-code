from pathlib import Path
import math
import subprocess
import numpy as np
from PIL import Image
import cv2

ROOT = Path(__file__).resolve().parent

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    h, w = original.shape[:2]
    # The annotation comprises the text and the black arc above the ray origin.
    annotation = np.zeros((h,w), dtype=bool)
    annotation[465:493, 554:650] = True
    gray = (original[:,:,0] == original[:,:,1]) & (original[:,:,1] == original[:,:,2])
    annotation[470:484, 483:515] = gray[470:484, 483:515] & (original[470:484,483:515,0] < 255)
    # Retain the vertical gray normal, which begins below the arc.
    annotation[482:,511:514] = False
    theta = math.asin(math.sin(math.radians(41.1)) / 1.627)
    start = np.array([512.,512.])
    end = np.array([512. + (h-1-512.)*math.tan(theta), h-1.])
    direction = (end-start)/np.linalg.norm(end-start)
    side = np.array([-direction[1],direction[0]])
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    command = ['ffmpeg','-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo',
               '-pix_fmt','rgb24','-s',f'{w}x{h}','-r','16','-i','-',
               '-an','-c:v','libx264','-crf','12','-preset','slow','-pix_fmt','yuv420p',
               '-movflags','+faststart',str(out/'video.mp4')]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE)
    for i in range(70):
        frame = original.copy()
        fade = min(1., i/10.)
        frame[annotation] = np.rint(original[annotation]*(1-fade)+255*fade).astype(np.uint8)
        progress = max(0., min(1., (i-10)/58.))
        if progress > 0:
            tip = start + progress*(end-start)
            mask = np.zeros((h,w),np.uint8)
            def point(p): return tuple(np.rint(p*256).astype(int))
            cv2.line(mask, point(start), point(tip), 255, 2, cv2.LINE_AA, shift=8)
            if np.linalg.norm(tip-start) > 20:
                for sign in [-1,1]:
                    wing = tip - direction*15 + side*sign*5
                    cv2.line(mask, point(wing), point(tip),255,2,cv2.LINE_AA,shift=8)
            alpha = mask[:,:,None]/255.
            frame = np.rint(frame*(1-alpha)+np.array([255,0,0])*alpha).astype(np.uint8)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait() != 0: raise RuntimeError('ffmpeg encoding failed')
    print(f'Refracted angle: {math.degrees(theta):.6f} degrees; endpoint: {end.tolist()}')

if __name__ == '__main__':
    main()
