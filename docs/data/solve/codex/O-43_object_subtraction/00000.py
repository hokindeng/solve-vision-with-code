from pathlib import Path
import subprocess
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    # The green, red, and yellow squares represent the cube objects.
    colors = [(0, 200, 0), (255, 0, 0), (255, 255, 0)]
    masks = [np.all(original == color, axis=2) for color in colors]
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    command = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
               '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16',
               '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'slow',
               '-crf', '0', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
               str(out / 'video.mp4')]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    for index in range(96):
        frame = original.copy()
        # Smooth, sequential dissolves distribute removal across the full video.
        for object_index, mask in enumerate(masks):
            progress = np.clip((index - object_index * 31) / 33.0, 0, 1)
            amount = progress * progress * (3 - 2 * progress)
            frame[mask] = np.rint(original[mask].astype(float) * (1 - amount)
                                  + 255 * amount).astype(np.uint8)
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    if process.wait():
        raise RuntimeError('ffmpeg failed')

if __name__ == '__main__':
    main()
