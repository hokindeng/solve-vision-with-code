from pathlib import Path
import numpy as np
from PIL import Image
import subprocess

ROOT = Path('/app')

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    orange = np.all(original == (255, 165, 0), axis=2)
    sy, sx = np.where(orange)
    dx, dy = sx - 255, sy - 459
    board = original.copy()
    board[orange] = (0, 255, 0)
    # Each leg has Manhattan-minimal length: 5, 8, 8, and 8 cells.
    corners = [(2,4), (2,8), (1,8), (3,8), (3,2), (8,2), (8,5), (8,4), (1,4)]
    route = [corners[0]]
    for target in corners[1:]:
        x,y = route[-1]
        while (x,y) != target:
            x += int(np.sign(target[0]-x))
            y += int(np.sign(target[1]-y))
            route.append((x,y))
    centers = np.array([(51+102*x, 51+102*y) for x,y in route], dtype=float)
    out = ROOT / 'output' / 'video.mp4'
    out.parent.mkdir(exist_ok=True)
    proc = subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-crf','18','-preset','medium','-pix_fmt','yuv420p','-movflags','+faststart',str(out)], stdin=subprocess.PIPE)
    for frame_idx in range(107):
        if frame_idx == 0:
            frame = original
        else:
            # Hold the initial and completed states briefly; travel the entire route.
            t = np.clip((frame_idx-3)/99, 0, 1)*(len(route)-1)
            i = min(int(t), len(route)-2)
            p = np.rint(centers[i]*(1-(t-i)) + centers[i+1]*(t-i)).astype(int)
            frame = board.copy()
            frame[p[1]+dy,p[0]+dx] = (255,165,0)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait():
        raise RuntimeError('ffmpeg failed')

if __name__ == '__main__':
    main()
