from pathlib import Path
import numpy as np
from PIL import Image
import subprocess

ROOT = Path('/app')
OUT = ROOT / 'output/video.mp4'

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    orange = np.all(original == (255, 165, 0), axis=2)
    yy, xx = np.where(orange)
    base = original.copy()
    base[orange] = (0, 255, 0)
    # Coordinates are (column, row), zero indexed. Each segment is Manhattan optimal.
    route = [(9, 3)]
    for destination, axes in [((3, 4), (1, 0)), ((6, 8), (1, 0)),
                              ((0, 0), (0, 1)), ((3, 3), (0, 1))]:
        pos = list(route[-1])
        for axis in axes:
            while pos[axis] != destination[axis]:
                pos[axis] += 1 if destination[axis] > pos[axis] else -1
                route.append(tuple(pos))
    # Pause briefly on each numbered waypoint, with motion spread across the video.
    keyframes = [(0, 0), (3, 0), (25, 7), (28, 7), (50, 14),
                 (53, 14), (96, 28), (99, 28), (118, 34), (121, 34)]
    OUT.parent.mkdir(exist_ok=True)
    proc = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
        '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16',
        '-i', '-', '-an', '-c:v', 'libx264', '-crf', '15', '-preset', 'slow',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(OUT)], stdin=subprocess.PIPE)
    for frame in range(122):
        distance = np.interp(frame, *zip(*keyframes))
        step = min(int(distance), len(route)-2)
        fraction = distance-step
        point = np.array(route[step])*(1-fraction)+np.array(route[step+1])*fraction
        # Grid lines are spaced 102 pixels apart; preserve the original sprite exactly.
        dx, dy = np.rint((point-np.array(route[0]))*102).astype(int)
        canvas = base.copy()
        canvas[yy+dy, xx+dx] = (255, 165, 0)
        if frame == 0:
            canvas = original
        proc.stdin.write(canvas.tobytes())
    proc.stdin.close()
    if proc.wait():
        raise RuntimeError('ffmpeg encoding failed')

if __name__ == '__main__':
    main()
