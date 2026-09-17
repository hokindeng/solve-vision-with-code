from pathlib import Path
import numpy as np
from PIL import Image
import subprocess

ROOT = Path('/app')
OUT = ROOT / 'output/video.mp4'

def smooth(t):
    t = np.clip(t, 0, 1)
    return t*t*(3-2*t)

def main():
    OUT.parent.mkdir(exist_ok=True)
    source = Image.open(ROOT/'first_frame.png').convert('RGB')
    base = np.array(source)
    # Reuse the original oval's exact outline and source color.
    oval = base[647:720, 109:252].copy()
    oval[np.all(oval == (114,191,38), axis=2)] = (114,229,210)
    oval = Image.fromarray(oval)
    command = ['ffmpeg','-y','-f','rawvideo','-pix_fmt','rgb24','-s','1024x1024',
               '-r','16','-i','-','-an','-c:v','libx264','-crf','12',
               '-preset','slow','-pix_fmt','yuv420p','-movflags','+faststart',str(OUT)]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    for n in range(60):
        frame = base.copy()
        for cx, start, end in [(518,7,27),(854,31,43)]:
            alpha = smooth((n-start)/(end-start))
            if alpha <= 0:
                continue
            scale = 1.0 if cx == 518 else 1.0 - .30*smooth((n-43)/12)
            patch = Image.new('RGB',(160,90),'white')
            shape = oval if scale == 1 else oval.resize((round(143*scale),round(73*scale)),Image.Resampling.LANCZOS)
            patch.paste(shape, ((160-shape.width)//2,(90-shape.height)//2))
            x,y = cx-80,638
            old = base[y:y+90,x:x+160]
            frame[y:y+90,x:x+160] = np.rint(old*(1-alpha)+np.array(patch)*alpha).astype(np.uint8)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError('Video encoding failed')

if __name__ == '__main__':
    main()
