from pathlib import Path
import subprocess
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent

def main():
    source = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    # Preserve the source's exact circular sprite, including its darker outline.
    mask = np.all(source == (255, 165, 0), axis=2) | np.all(source == (200, 120, 0), axis=2)
    sy, sx = np.where(mask)
    colors = source[sy, sx].copy()
    background = source.copy()
    background[mask] = (50, 200, 50)

    # Coordinates are zero-based (column, row). Each segment is Manhattan-shortest.
    targets = [(1, 7), (6, 3), (8, 7), (7, 9)]
    path = [(4, 9)]
    for tx, ty in targets:
        x, y = path[-1]
        while x != tx:
            x += 1 if tx > x else -1
            path.append((x, y))
        while y != ty:
            y += 1 if ty > y else -1
            path.append((x, y))
    assert len(path) == 24
    output = ROOT / 'output'
    output.mkdir(exist_ok=True)
    encoder = subprocess.Popen([
        'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
        '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024',
        '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '0',
        '-preset', 'medium', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
        str(output / 'video.mp4')
    ], stdin=subprocess.PIPE)
    for frame_index in range(102):
        progress = min(23.0, max(0.0, (frame_index - 4) / 4))
        edge = min(22, int(progress))
        fraction = progress - edge
        a, b = np.array(path[edge]), np.array(path[edge + 1])
        position = a + fraction * (b - a)
        dx, dy = np.rint((position - (4, 9)) * 93).astype(int)
        frame = background.copy()
        frame[sy + dy, sx + dx] = colors
        if frame_index == 0:
            assert np.array_equal(frame, source)
        encoder.stdin.write(frame.tobytes())
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError('Video encoding failed')

if __name__ == '__main__':
    main()
