from pathlib import Path
import subprocess
import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    mask = np.all(original == (0, 128, 128), axis=2).astype(np.uint8)
    count, labels, stats, centers = cv2.connectedComponentsWithStats(mask)
    assert count - 1 == 55
    background = original.copy()
    blocks = []
    for x, y, w, h, area in stats[1:]:
        assert (w, h, area) == (40, 40, 1600)
        left, top = int(x - 2), int(y - 2)
        sprite = original[top:top+44, left:left+44].copy()
        row = int((top + 22) // 51.2)
        target_top = int((row + 3) * 51.2) + 4
        assert target_top + 44 <= 1024
        blocks.append((left, top, target_top, sprite))
        background[top:top+44, left:left+44] = 255
    output = ROOT / 'output'
    output.mkdir(exist_ok=True)
    command = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
               '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024',
               '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'slow',
               '-crf', '0', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
               str(output / 'video.mp4')]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    for i in range(35):
        frame = background.copy()
        progress = i / 34
        for left, top, target_top, sprite in blocks:
            new_top = round(top + (target_top - top) * progress)
            frame[new_top:new_top+44, left:left+44] = sprite
        if i == 0:
            assert np.array_equal(frame, original)
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError('ffmpeg failed')

if __name__ == '__main__':
    main()
