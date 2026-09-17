from pathlib import Path
import numpy as np
from PIL import Image
import subprocess

ROOT = Path('/app')

def main():
    original = np.asarray(Image.open(ROOT / 'first_frame.png').convert('RGB')).copy()
    # Reuse the existing octagon verbatim, translated to the missing slot.
    completed = original.copy()
    completed[462:563, 820:923] = 255
    completed[462:563, 821:922] = original[462:563, 390:491]
    blank = original.copy()
    blank[462:563, 820:923] = 255
    out = ROOT / 'output' / 'video.mp4'
    out.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen([
        'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
        '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024',
        '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-qp', '0',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out)
    ], stdin=subprocess.PIPE)
    for i in range(25):
        t = i / 24
        if t <= .5:
            u = t * 2
            u = u*u*(3-2*u)
            frame = np.rint(original.astype(float)*(1-u) + blank.astype(float)*u).astype(np.uint8)
        else:
            u = (t-.5)*2
            u = u*u*(3-2*u)
            frame = np.rint(blank.astype(float)*(1-u) + completed.astype(float)*u).astype(np.uint8)
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    if process.wait():
        raise RuntimeError('ffmpeg failed')

if __name__ == '__main__':
    main()
