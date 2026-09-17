from pathlib import Path
import subprocess
import numpy as np
from PIL import Image

ROOT = Path('/app')

def main():
    source = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    # Reuse the existing green element to preserve its exact outline and color.
    target = source.copy()
    target[475:549, 852:926] = source[475:549, 529:603]
    question = np.zeros(source.shape[:2], dtype=bool)
    question[491:533, 876:902] = True
    question &= np.any(source < 255, axis=2)
    circle = np.zeros(source.shape[:2], dtype=bool)
    circle[475:549, 852:926] = np.any(target[475:549, 852:926] < 255, axis=2)
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    proc = subprocess.Popen([
        'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
        '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16', '-i', '-',
        '-an', '-c:v', 'libx264', '-preset', 'slow', '-crf', '0',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out / 'video.mp4')
    ], stdin=subprocess.PIPE)
    def ease(t):
        t = np.clip(t, 0, 1)
        return t * t * (3 - 2 * t)
    for i in range(25):
        frame = source.copy()
        erase = ease(i / 12)
        frame[question] = np.rint(source[question] * (1 - erase) + 255 * erase).astype(np.uint8)
        reveal = ease((i - 10) / 14)
        frame[circle] = np.rint(frame[circle] * (1 - reveal) + target[circle] * reveal).astype(np.uint8)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait():
        raise RuntimeError('Video encoding failed')

if __name__ == '__main__':
    main()
