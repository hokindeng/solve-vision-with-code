from pathlib import Path
import numpy as np
from PIL import Image
import subprocess

ROOT = Path('/app')

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    orange = np.all(original == (255, 165, 0), axis=2)
    ys, xs = np.where(orange)
    cx, cy = (xs.min()+xs.max())/2, (ys.min()+ys.max())/2
    background = original.copy()
    background[orange] = (0, 255, 0)
    # Each segment is Manhattan-shortest, with no diagonal motion.
    cells = [(4, 9)]
    for target in [(8,9), (8,8), (1,8), (1,5), (4,5), (4,6), (4,4), (1,4)]:
        x,y = cells[-1]
        while (x,y) != target:
            if x != target[0]: x += 1 if target[0] > x else -1
            else: y += 1 if target[1] > y else -1
            cells.append((x,y))
    centers = np.array([(51.5+102*x, 51.5+102*y) for x,y in cells])
    centers += np.array([cx,cy]) - centers[0]
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    proc = subprocess.Popen(['ffmpeg','-y','-f','rawvideo','-vcodec','rawvideo',
        '-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-',
        '-an','-c:v','libx264','-preset','medium','-crf','0','-pix_fmt','yuv420p',
        '-movflags','+faststart',str(out/'video.mp4')], stdin=subprocess.PIPE)
    for frame in range(92):
        # Three initial frames and four final frames leave time to read both states.
        progress = np.clip((frame-2)/85, 0, 1)*(len(cells)-1)
        index = min(int(progress),len(cells)-2)
        pos = centers[index]*(1-(progress-index))+centers[index+1]*(progress-index)
        dx,dy = np.rint(pos-[cx,cy]).astype(int)
        canvas = background.copy()
        canvas[ys+dy,xs+dx] = (255,165,0)
        if frame == 0: canvas = original
        proc.stdin.write(canvas.tobytes())
    proc.stdin.close()
    if proc.wait(): raise RuntimeError('ffmpeg failed')

if __name__ == '__main__':
    main()
