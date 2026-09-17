from pathlib import Path
import subprocess
import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    foreground = np.any(original != 255, axis=2).astype(np.uint8)
    _, labels = cv2.connectedComponents(foreground, connectivity=8)
    colors = {'red': (255, 0, 0), 'green': (0, 255, 0),
              'blue': (0, 0, 255), 'yellow': (255, 255, 0),
              'magenta': (255, 0, 255)}
    counts = {}
    for name, color in colors.items():
        # Count colored interiors individually, including touching objects.
        interior = np.all(original == color, axis=2).astype(np.uint8)
        counts[name] = cv2.connectedComponents(interior, connectivity=8)[0] - 1
    majority = max(counts, key=counts.get)
    print('Object counts:', counts, 'Majority:', majority)
    remove = np.zeros(original.shape[:2], dtype=bool)
    for name, color in colors.items():
        if name != majority:
            ids = np.unique(labels[np.all(original == color, axis=2)])
            remove |= np.isin(labels, ids[ids != 0])
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    command = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
               '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024',
               '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-preset',
               'slow', '-crf', '0', '-pix_fmt', 'yuv420p', '-movflags',
               '+faststart', str(out / 'video.mp4')]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE)
    for i in range(40):
        t = i / 39
        fade = t * t * (3 - 2 * t)
        frame = original.copy()
        frame[remove] = np.rint(original[remove].astype(float) * (1 - fade)
                                + 255 * fade).astype(np.uint8)
        assert np.array_equal(frame[~remove], original[~remove])
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError('ffmpeg encoding failed')

if __name__ == '__main__':
    main()
