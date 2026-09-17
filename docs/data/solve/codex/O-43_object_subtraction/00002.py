from pathlib import Path
import subprocess
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    cones = np.all(original == (0, 0, 255), axis=2)
    output = ROOT / 'output'
    output.mkdir(exist_ok=True)
    command = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
               '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16', '-i', '-',
               '-an', '-c:v', 'libx264', '-preset', 'slow', '-crf', '0',
               '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output / 'video.mp4')]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    for i in range(96):
        t = i / 95
        # Smoothly remove only the cone pixels over the entire clip.
        opacity_removed = t * t * (3 - 2 * t)
        frame = original.copy()
        frame[cones] = (round(255 * opacity_removed), round(255 * opacity_removed), 255)
        assert np.array_equal(frame[~cones], original[~cones])
        if i == 0:
            assert np.array_equal(frame, original)
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError('ffmpeg encoding failed')

if __name__ == '__main__':
    main()
