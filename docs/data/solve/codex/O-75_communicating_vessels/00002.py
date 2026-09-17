from pathlib import Path
import subprocess
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent

def main():
    source = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    fps, count = 16, 53
    # Equal tube areas: hydrostatic flow through the shared manifold makes
    # deviations from the conserved mean decay under viscous resistance.
    initial = np.array([59., 33., 19., 15.])
    equilibrium = initial.mean()
    k = 2.29
    spans = [(106,196), (340,430), (574,664), (808,898)]
    original_surfaces = np.array([289.,549.,684.,724.])
    final_surface = 879. - 10. * equilibrium
    oil = np.array([255.,227.,75.])
    rows = np.arange(229,879, dtype=float)
    cmd = ['ffmpeg','-y','-loglevel','error','-f','rawvideo',
           '-pix_fmt','rgb24','-s','1024x1024','-r',str(fps),'-i','-',
           '-an','-c:v','libx264','-preset','slow','-crf','16',
           '-pix_fmt','yuv420p','-movflags','+faststart',str(out/'video.mp4')]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for index in range(count):
        frame = source.copy()
        if index:
            u = index / (count-1)
            # A smooth terminal taper resolves the exponential's final tiny
            # residual at the end of this finite settling demonstration.
            remaining = np.exp(-k * 2.0 * u) * (1-u*u)**2
            surfaces = final_surface + (original_surfaces-final_surface)*remaining
            for (left,right), surface in zip(spans,surfaces):
                coverage = np.clip(rows + 1.0 - surface, 0., 1.)
                colors = np.rint(255. + coverage[:,None]*(oil-255.)).astype(np.uint8)
                frame[229:879,left:right] = colors[:,None,:]
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait():
        raise RuntimeError('ffmpeg failed')

if __name__ == '__main__':
    main()
