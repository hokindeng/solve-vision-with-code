from pathlib import Path
import subprocess
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    # Copy the existing unlit light, including its white surroundings.
    # All light centers are spaced by exactly 82 pixels.
    target = original.copy()
    regions = [(609, 660, 6, 15), (773, 824, 21, 31)]
    for left, right, _, _ in regions:
        shift = 82 if left == 609 else -82
        target[487:538, left:right] = original[487:538, left+shift:right+shift]
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    command = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
               '-pixel_format', 'rgb24', '-video_size', '1024x1024',
               '-framerate', '16', '-i', '-', '-an', '-c:v', 'libx264',
               '-preset', 'slow', '-crf', '0', '-pix_fmt', 'yuv420p',
               '-movflags', '+faststart', str(out / 'video.mp4')]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE)
    for frame_number in range(35):
        frame = original.copy()
        for left, right, start, end in regions:
            t = np.clip((frame_number-start)/(end-start), 0, 1)
            t = t*t*(3-2*t)
            region = np.s_[487:538, left:right]
            frame[region] = np.rint(original[region].astype(float)*(1-t) +
                                    target[region].astype(float)*t).astype(np.uint8)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError('ffmpeg failed')

if __name__ == '__main__':
    main()
