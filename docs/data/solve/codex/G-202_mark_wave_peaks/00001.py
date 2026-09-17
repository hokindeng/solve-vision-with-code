from pathlib import Path
import subprocess
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path('/app')
OUT = ROOT / 'output'

def main():
    OUT.mkdir(exist_ok=True)
    base = Image.open(ROOT / 'first_frame.png').convert('RGB')
    a = np.asarray(base)
    dark = np.all(a < 100, axis=2)
    # Interior crests are the minima of image-space y (wave-space maxima).
    columns = np.where(dark.any(axis=0))[0]
    ys = np.array([np.where(dark[:, x])[0].mean() for x in columns])
    candidates = []
    for i in range(12, len(columns)-12):
        if ys[i] == min(ys[i-12:i+13]):
            if not candidates or columns[i] - candidates[-1][0] > 30:
                candidates.append((int(columns[i]), float(ys[i])))
    peaks = [(x, round(y)) for x, y in candidates]
    scale = 4
    frames = []
    for frame_index in range(10):
        overlay = Image.new('RGBA', (base.width*scale, base.height*scale))
        draw = ImageDraw.Draw(overlay)
        for j, (x,y) in enumerate(peaks):
            progress = max(0, min(1, (frame_index - j*3)/3))
            if not progress:
                continue
            r = 20
            box = tuple(v*scale for v in (x-r,y-r,x+r,y+r))
            draw.arc(box, start=-90, end=-90+360*progress, fill=(235,0,0,255), width=3*scale)
            d = 4
            draw.ellipse(tuple(v*scale for v in (x-d,y-d,x+d,y+d)), fill=(235,0,0,255))
        overlay = overlay.resize(base.size, Image.Resampling.LANCZOS)
        result = Image.alpha_composite(base.convert('RGBA'), overlay).convert('RGB')
        frames.append(result)
    proc = subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-preset','slow','-crf','12','-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/'video.mp4')], stdin=subprocess.PIPE)
    for frame in frames:
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait():
        raise RuntimeError('ffmpeg failed')
    print('Peaks:', peaks)

if __name__ == '__main__':
    main()
