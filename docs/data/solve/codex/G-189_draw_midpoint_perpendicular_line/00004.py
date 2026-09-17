from pathlib import Path
import subprocess
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    output = ROOT / 'output' / 'video.mp4'
    output.parent.mkdir(exist_ok=True)
    cmd = ['ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
           '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16', '-i', '-',
           '-an', '-c:v', 'libx264', '-preset', 'slow', '-crf', '0',
           '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output)]
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    # The gray point at (793, 562) lies midway between y=460.5 and y=664.5.
    # Draw progressively from the inner edge of the upper rule to the lower rule.
    for i in range(50):
        frame = original.copy()
        if i:
            end = 463 + round(200 * i / 49)
            frame[463:end, 791:795] = (255, 0, 0)
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    errors = process.stderr.read()
    if process.wait():
        raise RuntimeError(errors.decode())

if __name__ == '__main__':
    main()
