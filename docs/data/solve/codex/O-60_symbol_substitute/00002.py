from pathlib import Path
import subprocess
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent

def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    # Confine all changes to the interior of the fourth symbol's panel.
    ys, xs = slice(465, 561), slice(412, 507)
    old = original[ys, xs].astype(float)
    target = np.full_like(original, 255)
    # Copy the reference symbol exactly, centered at (459, 512).
    target[478:547, 425:494] = original[43:112, 912:981]
    new = target[ys, xs].astype(float)
    output = ROOT / 'output' / 'video.mp4'
    output.parent.mkdir(exist_ok=True)
    process = subprocess.Popen([
        'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
        '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024',
        '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '18',
        '-preset', 'slow', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
        str(output)
    ], stdin=subprocess.PIPE)
    for i in range(52):
        t = i / 51
        frame = original.copy()
        if t <= 0.5:
            region = old + (255.0 - old) * smoothstep(t / 0.5)
        else:
            region = 255.0 + (new - 255.0) * smoothstep((t - 0.5) / 0.5)
        frame[ys, xs] = np.rint(region).astype(np.uint8)
        if i == 0:
            assert np.array_equal(frame, original)
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError('ffmpeg encoding failed')

if __name__ == '__main__':
    main()
