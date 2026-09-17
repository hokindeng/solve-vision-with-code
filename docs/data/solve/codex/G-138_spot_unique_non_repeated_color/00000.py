from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import cv2
import subprocess

ROOT = Path(__file__).resolve().parent

def main():
    base = Image.open(ROOT / 'first_frame.png').convert('RGB')
    a = np.array(base)
    # The singly occurring blue component is the triangle.
    mask = ((a[:,:,2] > a[:,:,0] + 25) & (a[:,:,0] > 60) & (a[:,:,0] < 150) & (a[:,:,1] < 160)).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    vertices = cv2.approxPolyDP(contour, 2, True).reshape(-1, 2)
    start = np.argmin(vertices[:,1])
    vertices = np.roll(vertices, -start, axis=0).astype(float)
    vertices = np.vstack([vertices, vertices[0]])
    lengths = np.linalg.norm(np.diff(vertices, axis=0), axis=1)
    output = ROOT / 'output'
    output.mkdir(exist_ok=True)
    proc = subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-crf','0','-preset','slow','-pix_fmt','yuv420p','-movflags','+faststart',str(output/'video.mp4')], stdin=subprocess.PIPE)
    scale = 4
    for frame in range(21):
        im = base.copy()
        if frame:
            remaining = lengths.sum() * frame / 20
            points = [tuple(vertices[0]*scale)]
            for i, length in enumerate(lengths):
                t = min(remaining / length, 1.0)
                points.append(tuple((vertices[i] + t*(vertices[i+1]-vertices[i]))*scale))
                remaining -= length
                if remaining <= 0:
                    break
            alpha = Image.new('L', (1024*scale,1024*scale))
            draw = ImageDraw.Draw(alpha)
            draw.line(points, fill=255, width=4*scale, joint='curve')
            alpha = alpha.resize(base.size, Image.Resampling.LANCZOS)
            im.paste((0,0,0), (0,0), alpha)
        proc.stdin.write(im.tobytes())
    proc.stdin.close()
    if proc.wait():
        raise RuntimeError('ffmpeg failed')

if __name__ == '__main__':
    main()
