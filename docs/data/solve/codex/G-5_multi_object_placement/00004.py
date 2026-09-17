from pathlib import Path
import subprocess
import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    foreground = np.any(original != 255, axis=2).astype(np.uint8)
    count, labels, stats, centers = cv2.connectedComponentsWithStats(foreground)
    objects = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] > 1000]
    markers = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] <= 1000]
    background = original.copy()
    sprites = []
    for obj in objects:
        mask = labels == obj
        color = original[mask][0]
        marker = next(i for i in markers if np.array_equal(original[labels == i][0], color))
        background[mask] = 255
        sprites.append((mask.astype(np.float32), color, centers[marker] - centers[obj]))
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    command = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-vcodec', 'rawvideo',
               '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16', '-i', '-', '-an',
               '-c:v', 'libx264', '-crf', '18', '-preset', 'slow', '-pix_fmt', 'yuv420p',
               '-movflags', '+faststart', str(out / 'video.mp4')]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    for frame_index in range(48):
        t = frame_index / 47
        frame = background.astype(np.float32)
        for mask, color, displacement in sprites:
            dx, dy = displacement * t
            alpha = cv2.warpAffine(mask, np.float32([[1, 0, dx], [0, 1, dy]]),
                                   (1024, 1024), flags=cv2.INTER_LINEAR)[..., None]
            frame = frame * (1 - alpha) + color * alpha
        rendered = np.rint(frame).astype(np.uint8)
        if frame_index == 0:
            assert np.array_equal(rendered, original)
        process.stdin.write(rendered.tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError('ffmpeg failed')

if __name__ == '__main__':
    main()
