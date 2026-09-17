from pathlib import Path
import subprocess
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path('/app')

def main():
    base = Image.open(ROOT / 'first_frame.png').convert('RGB')
    arr = np.array(base)
    # Locate the second-largest solid circle by its interior area.
    colors, counts = np.unique(arr.reshape(-1, 3), axis=0, return_counts=True)
    candidates = [(int(n), c) for c, n in zip(colors, counts)
                  if tuple(c) not in ((255,255,255), (0,0,0))]
    candidates.sort(key=lambda item: item[0], reverse=True)
    color = candidates[1][1]
    ys, xs = np.where(np.all(arr == color, axis=2))
    cx, cy = (xs.min()+xs.max())/2, (ys.min()+ys.max())/2
    radius = max(xs.max()-xs.min(), ys.max()-ys.min())/2 + 10
    output = ROOT / 'output'
    output.mkdir(exist_ok=True)
    proc = subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo',
        '-vcodec','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16',
        '-i','-','-an','-c:v','libx264','-crf','15','-pix_fmt','yuv420p',
        '-movflags','+faststart',str(output/'video.mp4')], stdin=subprocess.PIPE)
    for frame in range(40):
        im = base.copy()
        if frame:
            scale = 4
            mask = Image.new('L', (1024*scale, 1024*scale))
            draw = ImageDraw.Draw(mask)
            progress = min(frame / 37, 1)
            bounds = tuple(v*scale for v in (cx-radius, cy-radius, cx+radius, cy+radius))
            draw.arc(bounds, -90, -90+360*progress, fill=255, width=5*scale)
            mask = mask.resize(base.size, Image.Resampling.LANCZOS)
            im.paste((235, 25, 30), (0,0), mask)
        proc.stdin.write(im.tobytes())
    proc.stdin.close()
    if proc.wait():
        raise RuntimeError('ffmpeg failed')

if __name__ == '__main__':
    main()
