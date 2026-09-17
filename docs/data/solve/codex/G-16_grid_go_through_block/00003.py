from pathlib import Path
import numpy as np
from PIL import Image
import subprocess

ROOT = Path('/app')

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    # Retain the original raster for every stationary element, and lift only
    # the two-color orange agent out of the green start cell.
    mask = np.zeros(original.shape[:2], dtype=bool)
    area = original[700:791, 49:140]
    mask[700:791, 49:140] = ((area == (255,165,0)).all(axis=2) |
                                    (area == (200,120,0)).all(axis=2))
    sy, sx = np.where(mask)
    sprite = original[sy, sx].copy()
    background = original.copy()
    background[mask] = (50, 200, 50)

    # Cell coordinates (column, row). Every segment is a Manhattan shortest
    # path; the final leg travels above the purple/yellow row.
    cells = [(0,7), (1,7), (2,7), (3,7), (4,7), (4,6), (4,5),
             (4,4), (5,4), (6,4), (7,4), (8,4), (9,4), (9,3),
             (8,3), (7,3), (6,3), (5,3), (4,3), (3,3), (2,3), (1,3)]
    out = ROOT / 'output/video.mp4'
    out.parent.mkdir(parents=True, exist_ok=True)
    command = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
               '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024',
               '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '16',
               '-preset', 'slow', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out)]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE)
    for frame in range(94):
        progress = np.clip((frame - 4) / 4, 0, len(cells)-1)
        step = min(int(progress), len(cells)-2)
        blend = progress-step
        position = (1-blend)*np.array(cells[step]) + blend*np.array(cells[step+1])
        dx, dy = np.rint((position-np.array(cells[0]))*93).astype(int)
        canvas = background.copy()
        canvas[sy+dy, sx+dx] = sprite
        proc.stdin.write(canvas.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError('ffmpeg encoding failed')

if __name__ == '__main__':
    main()
