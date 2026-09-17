from pathlib import Path
import subprocess
import numpy as np
from PIL import Image

ROOT = Path('/app')

def smooth(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * t * (10 + t * (-15 + 6 * t))

def main():
    source = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    # Original, unscaled artwork; order is descending overall face size.
    specs = [
        ((433, 213, 122, 182), 230, 0.00, 0.96), # rabbit
        ((705, 603, 120, 98), 420, 0.00, 0.57),  # frog
        ((297, 693, 87, 83), 610, 0.43, 1.00),  # pig
        ((202, 58, 76, 72), 800, 0.00, 1.00),   # bear
    ]
    background = source.copy()
    sprites = []
    for (x, y, w, h), center, start, end in specs:
        art = source[y:y+h, x:x+w].copy()
        mask = np.any(art != 255, axis=2)
        background[y:y+h, x:x+w][mask] = 255
        sprites.append((art, mask, x, y, int(center-w/2), 944-h, start, end))
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    command = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
               '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024',
               '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-preset',
               'slow', '-crf', '0', '-pix_fmt', 'yuv420p', '-movflags',
               '+faststart', str(out / 'video.mp4')]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE)
    for index in range(40):
        frame = background.copy()
        t = index / 39
        for art, mask, x, y, dest_x, dest_y, start, end in sprites:
            amount = smooth((t-start)/(end-start))
            px = round(x + (dest_x-x)*amount)
            py = round(y + (dest_y-y)*amount)
            h, w = mask.shape
            frame[py:py+h, px:px+w][mask] = art[mask]
        if index == 0:
            assert np.array_equal(frame, source)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError('ffmpeg encoding failed')

if __name__ == '__main__':
    main()
