from pathlib import Path
import subprocess
import numpy as np
from PIL import Image

ROOT = Path('/app')

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    base = original.copy()
    # Restore the track and the position markers hidden by the two moving levers.
    marker = original[673:680, 201:208].copy()
    for old_x, marker_x in [(99, 122), (790, 818)]:
        base[648:705, old_x:old_x + 57] = 0
        base[673:680, marker_x-3:marker_x+4] = marker
    masks = [(original == color).all(axis=2) for color in [(128, 0, 128), (255, 192, 203)]]
    colors = [(128, 0, 128), (255, 192, 203), (75, 0, 130)]
    output = ROOT / 'output'
    output.mkdir(exist_ok=True)
    command = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-vcodec', 'rawvideo',
               '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16', '-i', '-', '-an',
               '-c:v', 'libx264', '-crf', '0', '-preset', 'medium', '-pix_fmt', 'yuv420p',
               '-movflags', '+faststart', str(output / 'video.mp4')]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE)
    for i in range(24):
        frame = base.copy()
        # Move the left lever through middle to right, then move the right lever.
        a = np.clip((i-1)/12, 0, 1)
        b = np.clip((i-12)/9, 0, 1)
        a = a*a*(3-2*a)
        b = b*b*(3-2*b)
        for x in [round(99 + 154*a), round(790 + 77*b)]:
            frame[648:705, x:x+57] = 128
        frame[masks[0]] = colors[0 if a < .25 else 1 if a < .75 else 2]
        frame[masks[1]] = colors[1 if b < .5 else 2]
        if i == 0:
            frame = original.copy()
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait():
        raise RuntimeError('ffmpeg failed')

if __name__ == '__main__':
    main()
