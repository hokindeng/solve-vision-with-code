from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import subprocess

ROOT = Path('/app')

def main():
    source = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    background = source[0, 0].copy()
    foreground = np.any(source != background, axis=2).astype(np.uint8)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(foreground)
    # Translate the original raster faces, keeping every stationary element.
    movements = [(209, 198, 482, 546), (199, 431, 484, -306),
                 (166, 633, 467, -241)]
    base = source.copy()
    sprites = []
    for left, top, dx, dy in movements:
        label = next(i for i, s in enumerate(stats)
                     if s[0] == left and s[1] == top)
        x, y, w, h, _ = stats[label]
        mask = labels[y:y+h, x:x+w] == label
        pixels = source[y:y+h, x:x+w].copy()
        base[labels == label] = background
        sprites.append((x, y, pixels, mask, dx, dy))
    output = ROOT / 'output'
    output.mkdir(exist_ok=True)
    process = subprocess.Popen([
        'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
        '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024',
        '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '15',
        '-preset', 'medium', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
        str(output / 'video.mp4')], stdin=subprocess.PIPE)
    for frame_index in range(64):
        t = frame_index / 63
        frame = base.copy()
        for x, y, pixels, mask, dx, dy in sprites:
            xx, yy = x + round(dx*t), y + round(dy*t)
            h, w = mask.shape
            region = frame[yy:yy+h, xx:xx+w]
            region[mask] = pixels[mask]
        if frame_index == 0:
            assert np.array_equal(frame, source)
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError('Video encoding failed')

if __name__ == '__main__':
    main()
