from pathlib import Path
import subprocess
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    # Extract the entire blue sprite, including its dark blue border.
    mask = (original[:, :, 2] > 100) & (original[:, :, 0] == 0) & (original[:, :, 1] == 0)
    sy, sx = np.nonzero(mask)
    sprite = original[sy, sx].copy()
    background = original.copy()
    background[mask] = (0, 128, 0)
    start = np.array([804., 466.])
    bottom = np.array([475., 753.])
    end = np.array([584., 215.])
    # Green's sole outgoing edge leads to bottom; bottom leads to red.
    output = ROOT / 'output'
    output.mkdir(exist_ok=True)
    process = subprocess.Popen([
        'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
        '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16', '-i', '-',
        '-an', '-c:v', 'libx264', '-crf', '0', '-preset', 'medium',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
        str(output / 'video.mp4')
    ], stdin=subprocess.PIPE)
    for i in range(30):
        if i <= 1:
            pos = start
        elif i <= 14:
            pos = start + (bottom - start) * ((i - 1) / 13)
        elif i <= 16:
            pos = bottom
        elif i <= 28:
            pos = bottom + (end - bottom) * ((i - 16) / 12)
        else:
            pos = end
        dx, dy = np.rint(pos - start).astype(int)
        frame = background.copy()
        frame[sy + dy, sx + dx] = sprite
        if i == 0:
            assert np.array_equal(frame, original)
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError('Video encoding failed')

if __name__ == '__main__':
    main()
