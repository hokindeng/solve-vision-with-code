"""Render viscously damped equalization, retaining the source illustration."""
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent

def main():
    source = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    orange = np.array([255, 191, 75], dtype=np.uint8)
    columns = [(81,171), (269,359), (457,547), (645,735), (833,923)]
    tops = [int(np.flatnonzero(np.all(source[:, a] == orange, axis=1))[0]) for a,b in columns]
    initial = np.array([29.,6.,19.,47.,37.])
    equilibrium = initial.mean()  # Equal cross-sections conserve sum of heights.
    target_y = 879 - equilibrium * 10
    n, fps, k = 63, 16, 1.66
    duration = (n-1)/fps
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    proc = subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pixel_format','rgb24',
        '-video_size','1024x1024','-framerate',str(fps),'-i','-','-an','-c:v','libx264',
        '-preset','slow','-crf','15','-pix_fmt','yuv420p','-movflags','+faststart',str(out/'video.mp4')], stdin=subprocess.PIPE)
    for index in range(n):
        frame = source.copy()
        if index:
            t = index/fps
            # Common bottom-manifold pressure produces dh/dt proportional
            # to mean(h)-h. Normalize the finite clip's tiny exponential tail.
            residual = (np.exp(-k*t)-np.exp(-k*duration))/(1-np.exp(-k*duration))
            for (a,b), top in zip(columns,tops):
                y = target_y + (top-target_y)*residual
                frame[230:879,a:b] = 255
                iy = int(np.floor(y))
                frame[iy+1:879,a:b] = orange
                coverage = 1-(y-iy)
                frame[iy,a:b] = np.rint(255*(1-coverage)+orange.astype(float)*coverage).astype(np.uint8)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait():
        raise RuntimeError('ffmpeg failed')

if __name__ == '__main__':
    main()
