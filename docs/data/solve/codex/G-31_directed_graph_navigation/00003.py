from pathlib import Path
import numpy as np
from PIL import Image
import subprocess

ROOT = Path(__file__).resolve().parent

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    mask = (original[:,:,2] > 128) & (original[:,:,0] == 0) & (original[:,:,1] == 0)
    ys, xs = np.where(mask)
    colors = original[ys, xs].copy()
    background = original.copy()
    background[ys, xs] = (0,128,0)
    start = np.array([458.,269.])
    middle = np.array([220.,707.])
    end = np.array([566.,559.])
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    proc = subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(out/'video.mp4')], stdin=subprocess.PIPE)
    for i in range(30):
        if i <= 1:
            pos = start
        elif i <= 15:
            t = (i-1)/14
            pos = start + t*(middle-start)
        elif i <= 28:
            t = (i-15)/13
            pos = middle + t*(end-middle)
        else:
            pos = end
        dx,dy = np.rint(pos-start).astype(int)
        frame = background.copy()
        frame[ys+dy, xs+dx] = colors
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError('Video encoding failed')

if __name__ == '__main__':
    main()
