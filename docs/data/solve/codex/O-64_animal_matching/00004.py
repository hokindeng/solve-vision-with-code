from pathlib import Path
import subprocess
import numpy as np
from PIL import Image
import cv2

ROOT = Path('/app')

def main():
    original = np.asarray(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    background = original[0, 0].copy()
    yy, xx = np.indices(original.shape[:2])
    foreground = (np.any(original != background, axis=2) & (xx < 500)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(foreground)
    # Translation vectors align each face with its corresponding silhouette.
    destinations = {
        289: (294, 576),  # cat
        92: (682, 277),   # dog
        176: (616, -257), # pig
        307: (292, -276), # rabbit
        110: (460, -537), # lion
    }
    clean = original.copy()
    sprites = []
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        if area < 100:
            continue
        mask = labels[y:y+h, x:x+w] == label
        pixels = original[y:y+h, x:x+w].copy()
        clean[y:y+h, x:x+w][mask] = background
        sprites.append((x, y, w, h, pixels, mask, destinations[int(x)]))
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    command = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
               '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024',
               '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '15',
               '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out / 'video.mp4')]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    for frame_index in range(64):
        progress = frame_index / 63
        frame = clean.copy()
        for x, y, w, h, pixels, mask, (dx, dy) in sprites:
            px = x + round(dx * progress)
            py = y + round(dy * progress)
            frame[py:py+h, px:px+w][mask] = pixels[mask]
        if frame_index == 0:
            assert np.array_equal(frame, original)
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError('Video encoding failed')

if __name__ == '__main__':
    main()
