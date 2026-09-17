from pathlib import Path
import math
import cv2
import numpy as np
from PIL import Image
import subprocess

ROOT = Path('/app')

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    base = original.copy()
    sprites = []
    for left in (151, 225):
        sprites.append(original[583:725, left:left+46].copy())
        base[583:723, left:left+46] = 255
        base[723:725, left:left+46] = (118,85,43)
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    proc = subprocess.Popen(['ffmpeg','-y','-f','rawvideo','-vcodec','rawvideo',
        '-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-',
        '-an','-c:v','libx264','-crf','16','-preset','slow','-pix_fmt','yuv420p',
        '-movflags','+faststart', str(out/'video.mp4')], stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    contact = math.asin((74-46)/142)
    def draw(frame, sprite, pivot, angle):
        # Rotate clockwise about the lower right corner, retaining the original lettering.
        c, s = math.cos(angle), math.sin(angle)
        matrix = np.array([[c,-s,pivot-46*c+142*s],
                           [s,c,725-46*s-142*c]], dtype=np.float64)
        rgb = cv2.warpAffine(sprite,matrix,(1024,1024),flags=cv2.INTER_LINEAR)
        alpha = cv2.warpAffine(np.full((142,46),255,np.uint8),matrix,(1024,1024),flags=cv2.INTER_LINEAR)
        # RGB is already premultiplied by the affine warp's black border.
        mask = alpha > 0
        a = alpha[mask,None].astype(np.float32)/255
        frame[mask] = np.clip(rgb[mask].astype(np.float32) + frame[mask]*(1-a),0,255).astype(np.uint8)
    for i in range(50):
        t = i/16
        if t <= .125:
            frame = original.copy()
        else:
            if t < .85:
                u = (t-.125)/(.85-.125)
                a1, a2 = contact*u*u, 0.0
            else:
                u = min(1., (t-.85)/1.95)
                # Increasing speed through the fall and a brief settling at the floor.
                progress = u*u*(3-2*u)
                a2 = math.pi/2*progress
                a1 = a2 + math.asin((74*math.cos(a2)-46)/142)
            frame = base.copy()
            if a2 == 0:
                frame[583:725,225:271] = sprites[1]
            else:
                draw(frame,sprites[1],271,a2)
            draw(frame,sprites[0],197,a1)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    err = proc.stderr.read()
    if proc.wait():
        raise RuntimeError(err.decode())

if __name__ == '__main__':
    main()
