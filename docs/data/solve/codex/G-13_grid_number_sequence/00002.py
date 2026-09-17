from pathlib import Path
import numpy as np
from PIL import Image
import subprocess

ROOT = Path('/app')
OUT = ROOT / 'output/video.mp4'

def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    first = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    mask = np.all(first == (255, 165, 0), axis=2)
    ys, xs = np.where(mask)
    base = first.copy()
    base[mask] = (0, 255, 0)
    # Grid coordinates are (column, row). Every leg is Manhattan-shortest.
    # Down before right keeps the first leg clear of the later number 3.
    cells = [(2, 4)]
    for target, axes in [((6, 9), (1, 0)), ((3, 2), (0, 1)),
                         ((6, 5), (0, 1)), ((9, 2), (0, 1))]:
        current = list(cells[-1])
        for axis in axes:
            while current[axis] != target[axis]:
                current[axis] += 1 if target[axis] > current[axis] else -1
                cells.append(tuple(current))
    centers = np.array([(51.5 + 102 * c, 51.5 + 102 * r) for c, r in cells])
    command = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
               '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024',
               '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '15',
               '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(OUT)]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE)
    for frame_index in range(113):
        if frame_index == 0:
            frame = first
        else:
            distance = min(frame_index / 109, 1) * (len(cells) - 1)
            segment = min(int(distance), len(cells) - 2)
            fraction = distance - segment
            position = centers[segment] * (1 - fraction) + centers[segment + 1] * fraction
            dx, dy = np.rint(position - centers[0]).astype(int)
            frame = base.copy()
            frame[ys + dy, xs + dx] = (255, 165, 0)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError('ffmpeg encoding failed')

if __name__ == '__main__':
    main()
