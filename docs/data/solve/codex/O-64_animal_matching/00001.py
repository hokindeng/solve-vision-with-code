from pathlib import Path
import subprocess
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent

def main():
    original = np.asarray(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    bg = original[0, 0].copy()
    # Crops include every pixel of each original animal. The displacement
    # aligns its head and appendages with the corresponding destination.
    animals = [
        ((110, 140, 219, 247), (493, 24)),       # bear
        ((269, 140, 432, 275), (498, 593)),      # frog
        ((104, 423, 223, 597), (494, 263)),      # rabbit
        ((271, 447, 396, 529), (515, 30)),       # dog
        ((110, 751, 225, 878), (681, -601)),     # cat
        ((312, 666, 495, 850), (254, -245)),     # lion
    ]
    background = original.copy()
    sprites = []
    for (x0, y0, x1, y1), displacement in animals:
        crop = original[y0:y1, x0:x1].copy()
        mask = np.any(crop != bg, axis=2)
        background[y0:y1, x0:x1][mask] = bg
        sprites.append((x0, y0, crop, mask, displacement))

    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    proc = subprocess.Popen([
        'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
        '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024',
        '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '16',
        '-preset', 'slow', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
        str(out / 'video.mp4')
    ], stdin=subprocess.PIPE)
    for frame_index in range(64):
        progress = frame_index / 63
        frame = background.copy()
        for x0, y0, crop, mask, (dx, dy) in sprites:
            x = x0 + round(dx * progress)
            y = y0 + round(dy * progress)
            h, w = mask.shape
            frame[y:y+h, x:x+w][mask] = crop[mask]
        if frame_index == 0:
            assert np.array_equal(frame, original)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError('ffmpeg failed')

if __name__ == '__main__':
    main()
