from pathlib import Path
import subprocess
import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    blue = np.all(original == (0, 123, 255), axis=2).astype(np.uint8)
    count, labels, stats, centers = cv2.connectedComponentsWithStats(blue)
    background = original.copy()
    blocks = []
    for x, y, w, h, area in stats[1:]:
        # Preserve each original block, including its two-pixel black outline.
        x, y, w, h = int(x)-2, int(y)-2, int(w)+4, int(h)+4
        blocks.append((x, y, original[y:y+h, x:x+w].copy()))
        background[y:y+h, x:x+w] = 255
    assert len(blocks) == 10
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    command = ['ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
               '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16', '-i', '-',
               '-an', '-c:v', 'libx264', '-preset', 'slow', '-crf', '10',
               '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out / 'video.mp4')]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    for frame_index in range(35):
        t = frame_index / 34
        progress = t*t*(3-2*t)
        dy = round((1024 / 9) * progress)
        frame = background.copy()
        for x, y, block in blocks:
            h, w = block.shape[:2]
            frame[y+dy:y+dy+h, x:x+w] = block
        if frame_index == 0:
            assert np.array_equal(frame, original)
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    error = process.stderr.read()
    if process.wait():
        raise RuntimeError(error.decode())

if __name__ == '__main__':
    main()
