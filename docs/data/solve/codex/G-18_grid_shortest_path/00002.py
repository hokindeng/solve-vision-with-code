from pathlib import Path
import numpy as np
from PIL import Image
import subprocess

ROOT = Path('/app')

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    mask = np.all(original == (128, 0, 128), axis=2)
    sy, sx = np.where(mask)
    background = original.copy()
    background[mask] = (255, 255, 0)
    # Six cells down, then one right: Manhattan distance seven.
    route = [(0, 102 * r) for r in range(7)] + [(102, 612)]
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    proc = subprocess.Popen([
        'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
        '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024',
        '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '0',
        '-preset', 'medium', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
        str(out / 'video.mp4')
    ], stdin=subprocess.PIPE)
    for frame_index in range(55):
        progress = 7 * frame_index / 54
        segment = min(int(progress), 6)
        alpha = progress - segment
        x0, y0 = route[segment]
        x1, y1 = route[segment + 1]
        dx = round(x0 + (x1 - x0) * alpha)
        dy = round(y0 + (y1 - y0) * alpha)
        frame = background.copy()
        frame[sy + dy, sx + dx] = (128, 0, 128)
        if frame_index == 0:
            assert np.array_equal(frame, original)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError('ffmpeg encoding failed')

if __name__ == '__main__':
    main()
