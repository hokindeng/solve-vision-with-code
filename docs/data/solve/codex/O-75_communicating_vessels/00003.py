from pathlib import Path
import subprocess
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    color = np.array([255, 191, 75], dtype=float)
    spans = [(106, 196), (340, 430), (574, 664), (808, 898)]
    tops = np.array([644., 659., 324., 759.])
    heights = np.array([23., 22., 55., 12.])
    equilibrium = heights.mean()  # equal cross sections conserve volume
    final_top = 879. - 10. * equilibrium
    fps, frames = 16, 58
    # Overdamped hydrostatic relaxation: dh/dt proportional to -(g/k)(h-mean).
    # Conductance incorporates channel geometry and the video time scale.
    g, k, conductance = 9.8, 1.87, 0.25
    duration = (frames - 1) / fps
    rate = g * conductance / k
    tail = np.exp(-rate * duration)
    proc = subprocess.Popen([
        'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
        '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024',
        '-r', str(fps), '-i', '-', '-an', '-c:v', 'libx264',
        '-preset', 'slow', '-crf', '16', '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart', str(OUT / 'video.mp4')
    ], stdin=subprocess.PIPE)
    for i in range(frames):
        frame = original.copy()
        if i:
            t = i / fps
            # Remove the tiny unresolved tail to finish at exact equilibrium.
            remaining = (np.exp(-rate * t) - tail) / (1. - tail)
            surfaces = final_top + (tops - final_top) * remaining
            for (left, right), initial, surface in zip(spans, tops, surfaces):
                start = int(np.floor(min(initial, final_top)))
                end = int(np.ceil(max(initial, final_top))) + 1
                rows = np.arange(start, end)
                coverage = np.clip(rows + 1. - surface, 0., 1.)
                pixels = np.rint(255. + coverage[:, None] * (color - 255.)).astype(np.uint8)
                frame[start:end, left:right] = pixels[:, None, :]
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait():
        raise RuntimeError('ffmpeg encoding failed')

if __name__ == '__main__':
    main()
