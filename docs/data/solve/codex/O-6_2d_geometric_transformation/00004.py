from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from scipy.optimize import minimize_scalar
import subprocess

ROOT = Path('/app')
def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    center = np.array([779., 359.])
    vertices = np.array([[779,359], [819,315], [885,376], [906,354], [950,395], [888,461]], dtype=float)
    def rotated(degrees):
        a = np.deg2rad(degrees)
        # Screen y increases downward: this matrix rotates counterclockwise.
        return (vertices-center) @ np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]]) + center
    yy, xx = np.where(np.all(original == (100,100,100), axis=2))
    target = np.column_stack((xx,yy))
    def error(degrees):
        q = rotated(degrees)
        edges = np.roll(q,-1,axis=0)-q
        delta = target[:,None,:]-q
        t = np.clip(np.sum(delta*edges,axis=2)/np.sum(edges*edges,axis=1),0,1)
        return np.mean(np.min(np.sum((delta-t[:,:,None]*edges)**2,axis=2),axis=1))
    angle = minimize_scalar(error,bounds=(195,205),method='bounded').x
    background = original.copy()
    polygon_pixels = np.all(original == (132,152,174),axis=2) | np.all(original == (50,50,50),axis=2)
    background[polygon_pixels] = (240,240,240)
    y,x = np.indices(original.shape[:2])
    marker = ((x-center[0])**2+(y-center[1])**2 <= 13**2) & (np.all(original==0,axis=2) | np.all(original==255,axis=2))
    output = ROOT / 'output'
    output.mkdir(exist_ok=True)
    command = ['ffmpeg','-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(output/'video.mp4')]
    proc = subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
    for i in range(70):
        if i == 0:
            frame = original.copy()
        else:
            t = np.clip((i-2)/65,0,1)
            progress = t*t*(3-2*t)
            if progress == 0:
                frame = original.copy()
            else:
                frame = background.copy()
                points = np.rint(rotated(angle*progress)).astype(np.int32)
                cv2.fillPoly(frame,[points],(132,152,174),lineType=cv2.LINE_8)
                cv2.polylines(frame,[points],True,(50,50,50),1,cv2.LINE_8)
                frame[marker] = original[marker]
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    errors = proc.stderr.read()
    if proc.wait() != 0:
        raise RuntimeError(errors.decode())
    print(f'Wrote {output / "video.mp4"}; rotation {angle:.4f} degrees')

if __name__ == '__main__':
    main()
