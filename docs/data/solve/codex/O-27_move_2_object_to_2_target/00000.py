from pathlib import Path
import cv2
import numpy as np
import subprocess

ROOT = Path(__file__).resolve().parent

def main():
    original = cv2.imread(str(ROOT / 'first_frame.png'))
    background = original.copy()
    # Source shapes are extracted verbatim, including their original edge pixels.
    specs = [(219, 282, 143, 143, 2, 208),
             (634, 286, 136, 129, 156, -182)]
    sprites = []
    for x, y, w, h, dx, dy in specs:
        patch = original[y:y+h, x:x+w].copy()
        mask = np.any(patch != (220, 220, 220), axis=2)
        background[y:y+h, x:x+w][mask] = (220, 220, 220)
        sprites.append((x, y, w, h, dx, dy, patch, mask))
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    command = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
               '-vcodec', 'rawvideo', '-pix_fmt', 'bgr24', '-s', '1024x1024',
               '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '0',
               '-preset', 'medium', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
               str(out / 'video.mp4')]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    for frame_number in range(35):
        t = frame_number / 34
        frame = background.copy()
        for x, y, w, h, dx, dy, patch, mask in sprites:
            px, py = x + round(dx*t), y + round(dy*t)
            frame[py:py+h, px:px+w][mask] = patch[mask]
        if frame_number == 0:
            assert np.array_equal(frame, original)
        encoder.stdin.write(frame.tobytes())
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError('ffmpeg encoding failed')

if __name__ == '__main__':
    main()
