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
    annotation = np.zeros((h,w), bool)
    annotation[468:492, 557:650] = np.any(original[468:492,557:650] != 255, axis=2)
    region = original[470:483,480:515]
    annotation[470:483,480:515] = np.all(region == 0, axis=2)
    angle = math.asin(math.sin(math.radians(41.7))/1.387)
    start = np.array([512., 512.])
    end = np.array([512.+(1023.-512.)*math.tan(angle), 1023.])
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    cmd = ['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pixel_format','rgb24','-video_size','1024x1024','-framerate','16','-i','-','-an','-c:v','libx264','-preset','slow','-crf','10','-pix_fmt','yuv420p','-movflags','+faststart',str(out/'video.mp4')]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(70):
        frame = original.copy()
        fade = min(1., i/12.)
        frame[annotation] = np.round(original[annotation]*(1-fade)+255*fade).astype(np.uint8)
        progress = np.clip((i-10)/57., 0., 1.)
        if progress > 0:
            tip = start + progress*(end-start)
            mask = np.zeros((h,w), np.uint8)
            cv2.line(mask, tuple(np.round(start*256).astype(int)), tuple(np.round(tip*256).astype(int)), 255, 2, cv2.LINE_AA, shift=8)
            alpha = mask.astype(float)[...,None]/255
            frame = np.round(frame*(1-alpha)+np.array([255,0,0])*alpha).astype(np.uint8)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait():
        raise RuntimeError('ffmpeg failed')
    Image.fromarray(frame).save(out/'last_frame.png')
    print(f'Refracted angle: {math.degrees(angle):.4f} degrees; endpoint: {end}')

if __name__ == '__main__':
    main()
