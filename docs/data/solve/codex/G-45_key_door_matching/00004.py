from pathlib import Path
import numpy as np
from PIL import Image
import subprocess

ROOT = Path('/app')
OUT = ROOT / 'output/video.mp4'

def along(points, fraction):
    points = np.asarray(points, dtype=float)
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    distance = fraction * lengths.sum()
    for a, b, length in zip(points[:-1], points[1:], lengths):
        if distance <= length:
            return np.rint(a + (b-a) * distance / length).astype(int)
        distance -= length
    return points[-1].astype(int)

def main():
    source = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    green = np.all(source == (0,255,0), axis=2)
    gy,gx = np.where(green)
    offsets_y, offsets_x = gy-139, gx-139
    background = source.copy()
    background[green] = 255
    yy,xx = np.indices(source.shape[:2])
    key = np.all(source == (255,255,0), axis=2) & (xx < 600) & (yy < 400)
    before = [(139,139),(139,325),(325,325),(325,511),(511,511),(511,325)]
    after = [(511,325),(697,325),(697,511),(883,511),(883,791)]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    command = ['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24',
               '-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264',
               '-crf','0','-preset','medium','-pix_fmt','yuv420p','-movflags','+faststart',str(OUT)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    for i in range(52):
        if i == 0:
            frame = source.copy()
        else:
            frame = background.copy()
            if i < 26:
                x,y = along(before, i/26)
            else:
                frame[key] = 255
                x,y = along(after, min(1,(i-26)/23))
            # Check the full circle footprint stays within the white corridor.
            assert not np.any(np.all(background[y+offsets_y,x+offsets_x] == 0, axis=1))
            frame[y+offsets_y,x+offsets_x] = (0,255,0)
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError('ffmpeg encoding failed')

if __name__ == '__main__':
    main()
