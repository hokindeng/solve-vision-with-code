from pathlib import Path
import subprocess
import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    height, width = original.shape[:2]
    mask = ((original[:, :, 0] == 128) & (original[:, :, 1] == 0) & (original[:, :, 2] == 128)).astype(np.uint8)
    count, labels, stats, centers = cv2.connectedComponentsWithStats(mask)
    assert count - 1 == 27
    background = original.copy()
    blocks = []
    pitch = width / 9
    for x, y, w, h, area in stats[1:]:
        # Each solid purple interior has a two-pixel black border.
        x, y, w, h = int(x)-2, int(y)-2, int(w)+4, int(h)+4
        sprite = original[y:y+h, x:x+w].copy()
        background[y:y+h, x:x+w] = 255
        col = int((x + w/2) / pitch)
        positions = [x] + [int((col + step + .08) * pitch) for step in (1, 2)]
        assert positions[-1] + w <= width
        blocks.append((y, sprite, positions))
    output = ROOT / 'output'
    output.mkdir(exist_ok=True)
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-vcodec', 'rawvideo',
           '-pix_fmt', 'rgb24', '-s', f'{width}x{height}', '-r', '16', '-i', '-',
           '-an', '-c:v', 'libx264', '-preset', 'slow', '-crf', '0',
           '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output / 'video.mp4')]
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for frame_index in range(35):
        phase = frame_index / 17
        step = min(int(phase), 1)
        t = phase - step
        t = t*t*(3-2*t)
        frame = background.copy()
        for y, sprite, positions in blocks:
            x = round(positions[step] * (1-t) + positions[step+1] * t)
            h, w = sprite.shape[:2]
            frame[y:y+h, x:x+w] = sprite
        if frame_index == 0:
            assert np.array_equal(frame, original)
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    assert process.wait() == 0

if __name__ == '__main__':
    main()
