from pathlib import Path
import numpy as np
from PIL import Image
import imageio.v2 as imageio

ROOT = Path('/app')

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    mask = np.all(original == (255, 165, 0), axis=2) | np.all(original == (200, 120, 0), axis=2)
    ys, xs = np.where(mask)
    colors = original[ys, xs].copy()
    background = original.copy()
    background[ys, xs] = (50, 200, 50)
    # Cell centers are 93 pixels apart. Every segment follows grid edges.
    cells = [(4, 0)]
    def move(dx, dy, count):
        for _ in range(count):
            x, y = cells[-1]
            cells.append((x + dx, y + dy))
    move(0, 1, 5)
    move(-1, 0, 1)  # Blue (3, 5)
    move(-1, 0, 1)  # Purple (2, 5)
    move(0, 1, 2)
    move(1, 0, 3)   # Pink (5, 7)
    move(1, 0, 3)
    move(0, -1, 5)  # Yellow (8, 2)
    move(0, -1, 1)
    move(-1, 0, 2)  # Red (6, 1)
    assert len(cells) == 24
    offsets = [(0, 0)] * 4
    for a, b in zip(cells, cells[1:]):
        for step in range(1, 5):
            t = step / 4
            offsets.append((round(93 * (a[0] + t * (b[0] - a[0]) - 4)),
                            round(93 * (a[1] + t * (b[1] - a[1])))))
    offsets.extend([offsets[-1]] * 6)
    assert len(offsets) == 102
    (ROOT / 'output').mkdir(exist_ok=True)
    with imageio.get_writer(ROOT / 'output/video.mp4', fps=16, codec='libx264',
                            pixelformat='yuv420p', macro_block_size=None,
                            ffmpeg_params=['-crf', '18']) as writer:
        for dx, dy in offsets:
            frame = background.copy()
            frame[ys + dy, xs + dx] = colors
            writer.append_data(frame)

if __name__ == '__main__':
    main()
