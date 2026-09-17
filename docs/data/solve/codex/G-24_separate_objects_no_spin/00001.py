from pathlib import Path
import numpy as np
from PIL import Image
import subprocess

ROOT = Path('/app')

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    background = original.copy()
    # Tight bounds include each shape's original outline and antialiased edge.
    specs = [(52,352,289,589,618), (309,412,412,529,260), (370,588,460,672,374)]
    sprites = []
    for x0,y0,x1,y1,dx in specs:
        patch = original[y0:y1,x0:x1].copy()
        mask = np.any(patch != 255, axis=2)
        sprites.append((x0,y0,patch,mask,dx))
        background[y0:y1,x0:x1][mask] = 255
    # Retain all original target markings, including where a moving shape passes.
    fixed = np.any(background != 255, axis=2)
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    process = subprocess.Popen([
        'ffmpeg','-y','-loglevel','error','-f','rawvideo','-pixel_format','rgb24',
        '-video_size','1024x1024','-framerate','16','-i','-', '-an',
        '-c:v','libx264','-crf','16','-preset','slow','-pix_fmt','yuv420p',
        '-movflags','+faststart',str(out/'video.mp4')], stdin=subprocess.PIPE)
    for i in range(30):
        t = i/29
        progress = t*t*(3-2*t)
        frame = background.copy()
        for x,y,patch,mask,dx in sprites:
            xx = x + round(dx*progress)
            view = frame[y:y+patch.shape[0], xx:xx+patch.shape[1]]
            view[mask] = patch[mask]
        frame[fixed] = background[fixed]
        if i == 0:
            frame = original.copy()
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    if process.wait():
        raise RuntimeError('ffmpeg encoding failed')

if __name__ == '__main__':
    main()
