from pathlib import Path
import subprocess
import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    foreground = np.any(original != 255, axis=2)
    n, labels = cv2.connectedComponents(foreground.astype(np.uint8), connectivity=8)
    remove = np.zeros(foreground.shape, dtype=bool)
    # Components include the original black outlines, so no outline is left behind.
    for label in range(1, n):
        mask = labels == label
        pixels = original[mask]
        has_red = np.any(np.all(pixels == (255, 0, 0), axis=1))
        if not has_red:
            remove |= mask
    # The triangle overlaps a red square and consequently shares its component.
    for y in range(602, 667):
        if y == 602:
            left, right = 447, 447
        elif y == 603:
            left, right = 446, 447
        elif y == 666:
            left, right = 415, 479
        else:
            xs = np.where(np.all(original[y] == (0, 255, 0), axis=1))[0]
            xs = xs[(xs >= 415) & (xs <= 479)]
            left, right = int(xs.min()) - 1, int(xs.max()) + 1
        remove[y, left:right + 1] = True
    target = original.copy()
    target[remove] = 255
    # Reconstruct only the occluded portion of the stationary square.
    square = np.full_like(original, 255)
    square[612:643, 456:487] = 0
    square[614:641, 458:485] = (255, 0, 0)
    square_mask = np.zeros_like(remove)
    square_mask[612:643, 456:487] = True
    reveal = remove & square_mask
    target[reveal] = square[reveal]
    assert np.array_equal(original[~remove], target[~remove])
    output = ROOT / 'output'
    output.mkdir(exist_ok=True)
    command = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
               '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16', '-i', '-',
               '-an', '-c:v', 'libx264', '-preset', 'slow', '-crf', '0',
               '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output / 'video.mp4')]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE)
    for index in range(40):
        t = index / 39
        alpha = t*t*(3-2*t)
        frame = original.copy()
        frame[remove] = np.rint(original[remove].astype(float)*(1-alpha) + target[remove]*alpha).astype(np.uint8)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError('ffmpeg failed')

if __name__ == '__main__':
    main()
