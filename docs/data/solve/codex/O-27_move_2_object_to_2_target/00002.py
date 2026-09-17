from pathlib import Path
import numpy as np
from PIL import Image
import imageio.v2 as imageio

ROOT = Path(__file__).resolve().parent

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    yy, xx = np.indices(original.shape[:2])
    pink = (original[:, :, 0] > original[:, :, 1]) & (original[:, :, 2] > original[:, :, 1]) & (yy >= 175)
    green = (original[:, :, 1] > original[:, :, 0]) & (yy >= 633)
    background = original.copy()
    background[pink | green] = (220, 220, 220)
    objects = []
    for mask, displacement in [(pink, (129, -140)), (green, (309, -215))]:
        y, x = np.nonzero(mask)
        objects.append((y, x, original[y, x].copy(), displacement))
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    with imageio.get_writer(out / 'video.mp4', fps=16, codec='libx264', pixelformat='yuv420p', macro_block_size=None, ffmpeg_params=['-crf', '0', '-preset', 'medium']) as writer:
        for frame_index in range(35):
            frame = background.copy()
            progress = frame_index / 34
            for y, x, pixels, (dx, dy) in objects:
                frame[y + round(progress * dy), x + round(progress * dx)] = pixels
            if frame_index == 0:
                assert np.array_equal(frame, original)
            writer.append_data(frame)

if __name__ == '__main__':
    main()
