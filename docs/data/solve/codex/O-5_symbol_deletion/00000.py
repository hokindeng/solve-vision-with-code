from pathlib import Path
import subprocess
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent

def main():
    source = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    output = ROOT / 'output' / 'video.mp4'
    output.parent.mkdir(parents=True, exist_ok=True)
    # The deletion target is the rightmost tile. All pixels outside its
    # original bounds are copied verbatim throughout the animation.
    region = np.s_[446:579, 648:781]
    original = source[region].astype(np.float64)
    command = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
               '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024',
               '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '0',
               '-preset', 'medium', '-pix_fmt', 'yuv420p', '-movflags',
               '+faststart', str(output)]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    for i in range(46):
        frame = source.copy()
        # Initial target-identification hold, followed by gradual deletion,
        # and a final hold showing the unchanged three-symbol sequence.
        t = np.clip((i - 7) / 32.0, 0.0, 1.0)
        amount = t * t * (3.0 - 2.0 * t)
        frame[region] = np.rint(original * (1.0 - amount) + 255 * amount).astype(np.uint8)
        encoder.stdin.write(frame.tobytes())
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError('Video encoding failed')

if __name__ == '__main__':
    main()
