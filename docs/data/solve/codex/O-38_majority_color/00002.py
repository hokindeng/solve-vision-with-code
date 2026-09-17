from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import subprocess

ROOT = Path('/app')

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    nonwhite = np.any(original != 255, axis=2).astype(np.uint8)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(nonwhite, 8)
    remove = np.zeros(original.shape[:2], dtype=bool)
    # Each isolated minority object includes its complete black outline.
    for color in [(0,255,0), (0,0,255), (255,255,0)]:
        ids = np.unique(labels[np.all(original == color, axis=2)])
        remove |= np.isin(labels, ids[ids != 0])
    # The magenta circle touches a red triangle. Select its circular footprint
    # independently so no red pixels are modified.
    y, x = np.indices(remove.shape)
    circle = (x - 931)**2 + (y - 156)**2 <= 29.7**2
    black = np.all(original == 0, axis=2)
    magenta = np.all(original == (255,0,255), axis=2)
    remove |= magenta | (circle & black)
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    command = ['ffmpeg','-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo',
               '-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-',
               '-an','-c:v','libx264','-preset','slow','-crf','12',
               '-pix_fmt','yuv420p','-movflags','+faststart',str(out/'video.mp4')]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE)
    for i in range(40):
        t = i / 39
        alpha = t*t*(3-2*t)
        frame = original.copy()
        frame[remove] = np.rint(original[remove].astype(float)*(1-alpha)+255*alpha).astype(np.uint8)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError('ffmpeg failed')

if __name__ == '__main__':
    main()
