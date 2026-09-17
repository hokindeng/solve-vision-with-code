from pathlib import Path
import numpy as np
from PIL import Image
import imageio.v2 as imageio

ROOT = Path('/app')

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    background = original.copy()
    # Regions contain only each original animal and the white canvas.
    regions = [(410, 795, 520, 920), (180, 390, 320, 510), (740, 390, 890, 595)]
    animals = []
    for left, top, right, bottom in regions:
        region = original[top:bottom, left:right]
        ys, xs = np.where(np.any(region != 255, axis=2))
        x0, y0 = left + xs.min(), top + ys.min()
        x1, y1 = left + xs.max() + 1, top + ys.max() + 1
        sprite = original[y0:y1, x0:x1].copy()
        mask = np.any(sprite != 255, axis=2)
        animals.append((sprite, mask, x0, y0))
        background[y0:y1, x0:x1] = 255
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    # Equal center spacing, ascending size; feet of the faces meet the baseline.
    targets = [(260 - a[0].shape[1] // 2, 944 - a[0].shape[0]) for a in animals]
    targets[1] = (512 - animals[1][0].shape[1] // 2, 944 - animals[1][0].shape[0])
    targets[2] = (764 - animals[2][0].shape[1] // 2, 944 - animals[2][0].shape[0])
    with imageio.get_writer(out / 'video.mp4', fps=16, codec='libx264', pixelformat='yuv420p', macro_block_size=None, quality=10, ffmpeg_params=['-crf', '0']) as writer:
        for i in range(40):
            t = i / 39
            progress = t * t * (3 - 2 * t)
            frame = background.copy()
            for (sprite, mask, x0, y0), (tx, ty) in zip(animals, targets):
                x = round(x0 + (tx - x0) * progress)
                y = round(y0 + (ty - y0) * progress)
                h, w = sprite.shape[:2]
                frame[y:y+h, x:x+w][mask] = sprite[mask]
            if i == 0:
                assert np.array_equal(frame, original)
            writer.append_data(frame)

if __name__ == '__main__':
    main()
