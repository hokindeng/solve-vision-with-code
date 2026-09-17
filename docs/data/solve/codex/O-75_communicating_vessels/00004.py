"""Render viscously damped communicating vessels using the supplied artwork."""
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output' / 'video.mp4'
FPS, FRAMES = 16, 125
K = 0.82

def main():
    original = np.asarray(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    ink = np.array([165., 75., 75.])
    # Equal bore tubes, with their exact interior coordinates in the artwork.
    bores = [(223, 313), (457, 547), (691, 781)]
    initial_surfaces = np.array([294., 559., 734.])
    # The artwork is schematic: conserving its filled area places the common
    # surface at the mean of the three original pixel coordinates. In physical
    # units the conserved mean is (58 + 32 + 14)/3 = 34 2/3 cm.
    equilibrium = initial_surfaces.mean()
    laplacian = np.array([[1., -1., 0.], [-1., 2., -1.], [0., -1., 1.]])
    rates, modes = np.linalg.eigh(laplacian)
    amplitudes = modes.T @ (initial_surfaces - equilibrium)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    command = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
               '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024',
               '-r', str(FPS), '-i', '-', '-an', '-c:v', 'libx264',
               '-preset', 'slow', '-crf', '15', '-pix_fmt', 'yuv420p',
               '-movflags', '+faststart', str(OUT)]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    for index in range(FRAMES):
        frame = original.copy()
        if index:
            t = index / (FRAMES - 1)
            # Overdamped hydrostatic network: dh/dt = -k L h. Remove the
            # tiny finite-duration exponential tail so the final frame settles
            # exactly, while retaining a slow, monotonic viscous redistribution.
            decay = np.zeros(3)
            for j in (1, 2):
                endpoint = np.exp(-K * rates[j] * 7.75)
                decay[j] = (np.exp(-K * rates[j] * 7.75 * t) - endpoint) / (1 - endpoint)
            surfaces = equilibrium + modes @ (amplitudes * decay)
            for (left, right), surface in zip(bores, surfaces):
                rows = np.arange(229, 875, dtype=float)
                coverage = np.clip(rows + 1 - surface, 0, 1)
                colors = np.rint(255 * (1 - coverage[:, None]) + ink * coverage[:, None]).astype(np.uint8)
                frame[229:875, left:right] = colors[:, None, :]
        encoder.stdin.write(frame.tobytes())
    encoder.stdin.close()
    if encoder.wait():
        raise RuntimeError('Video encoding failed')

if __name__ == '__main__':
    main()
