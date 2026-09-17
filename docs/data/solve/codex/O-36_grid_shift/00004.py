from pathlib import Path
import subprocess
import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    green = np.all(original == (0, 255, 0), axis=2).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(green)
    assert count - 1 == 82
    background = original.copy()
    blocks = []
    for x, y, w, h, area in stats[1:]:
        # The source squares have a two-pixel black border.
        x, y, w, h = int(x)-2, int(y)-2, int(w)+4, int(h)+4
        tile = original[y:y+h, x:x+w].copy()
        blocks.append((x, y, w, h, tile))
        background[y:y+h, x:x+w] = 255
        assert y >= 64
    output = ROOT / 'output'
    output.mkdir(exist_ok=True)
    command = ['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo',
               '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16',
               '-i', '-', '-an', '-c:v', 'libx264', '-crf', '0',
               '-preset', 'medium', '-pix_fmt', 'yuv420p',
               '-movflags', '+faststart', str(output / 'video.mp4')]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE)
    for frame_index in range(35):
        shift = round(64 * frame_index / 34)
        frame = background.copy()
        for x, y, w, h, tile in blocks:
            frame[y-shift:y-shift+h, x:x+w] = tile
        if frame_index == 0:
            assert np.array_equal(frame, original)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    assert proc.wait() == 0

if __name__ == '__main__':
    main()
