from pathlib import Path
import subprocess
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent

def main():
    source = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    box_mask = np.all(source == (70, 140, 70), axis=2)
    ys, xs = np.where(box_mask)
    background = source.copy()
    background[box_mask] = (255, 255, 255)
    # Preserve the original attention outline exactly, translating it to the circle.
    circle = np.all(source == (94, 156, 86), axis=2)
    cy, cx = np.where(circle)
    shift_x = round((cx.min() + cx.max() - xs.min() - xs.max()) / 2)
    shift_y = round((cy.min() + cy.max() - ys.min() - ys.max()) / 2)
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    command = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
               '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024',
               '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-preset',
               'slow', '-crf', '0', '-pix_fmt', 'yuv420p', '-movflags',
               '+faststart', str(out / 'video.mp4')]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    for i in range(25):
        t = i / 24
        progress = t * t * (3 - 2 * t)
        dx, dy = round(shift_x * progress), round(shift_y * progress)
        frame = background.copy()
        frame[ys + dy, xs + dx] = (70, 140, 70)
        if i == 0:
            assert np.array_equal(frame, source)
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError('ffmpeg failed')

if __name__ == '__main__':
    main()
