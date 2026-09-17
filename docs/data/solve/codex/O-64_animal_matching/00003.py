from pathlib import Path
import numpy as np
from PIL import Image
import imageio.v2 as imageio

ROOT = Path(__file__).resolve().parent

def main():
    original = np.asarray(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    background = original.copy()
    bg = original[0, 0]
    # Bounds contain the complete original sprites, including their dark edges.
    animals = [
        ((84, 204, 201, 381), (706, 407)),   # rabbit -> lower right
        ((311, 262, 412, 361), (296, -25)), # fox -> upper left outline
        ((112, 700, 283, 805), (651, -450)),# dog -> upper right
        ((294, 593, 435, 749), (293, 43)),  # cat -> lower left outline
    ]
    sprites = []
    for (x0, y0, x1, y1), displacement in animals:
        pixels = original[y0:y1, x0:x1].copy()
        mask = np.any(pixels != bg, axis=2)
        background[y0:y1, x0:x1][mask] = bg
        sprites.append((x0, y0, pixels, mask, displacement))
    (ROOT / 'output').mkdir(exist_ok=True)
    with imageio.get_writer(ROOT / 'output/video.mp4', fps=16,
                            codec='libx264', pixelformat='yuv420p',
                            macro_block_size=None, ffmpeg_params=['-crf', '10']) as video:
        for frame_index in range(64):
            frame = background.copy()
            progress = frame_index / 63
            for x0, y0, pixels, mask, (dx, dy) in sprites:
                x = x0 + round(dx * progress)
                y = y0 + round(dy * progress)
                h, w = mask.shape
                frame[y:y+h, x:x+w][mask] = pixels[mask]
            if frame_index == 0:
                assert np.array_equal(frame, original)
            video.append_data(frame)

if __name__ == '__main__':
    main()
