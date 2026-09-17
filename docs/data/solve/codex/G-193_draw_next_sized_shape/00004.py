from pathlib import Path
import subprocess
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path('/app')

def main():
    base = Image.open(ROOT / 'first_frame.png').convert('RGB')
    arr = np.array(base)
    # Repeat the third diamond: small, large, medium; small, large, medium.
    target = np.array(base)
    source = arr[480:545, 405:471]
    target[480:545, 853:919] = source
    shape = np.any(target != arr, axis=2)
    vertices = [(886,480),(918,512),(886,544),(853,512),(886,480)]
    (ROOT / 'output').mkdir(exist_ok=True)
    proc = subprocess.Popen([
        'ffmpeg','-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo',
        '-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-',
        '-an','-c:v','libx264','-preset','slow','-crf','0','-pix_fmt','yuv420p',
        '-movflags','+faststart',str(ROOT / 'output/video.mp4')
    ], stdin=subprocess.PIPE)
    for i in range(60):
        frame = arr.copy()
        reveal = np.zeros(shape.shape, dtype=bool)
        if i >= 8:
            # Trace the four edges in order, then fill from top to bottom.
            progress = min(4.0, (i-8)/26*4)
            line = Image.new('L',base.size,0)
            draw = ImageDraw.Draw(line)
            for edge in range(4):
                t = max(0.0,min(1.0,progress-edge))
                if t > 0:
                    a,b = vertices[edge],vertices[edge+1]
                    end = (round(a[0]+(b[0]-a[0])*t),round(a[1]+(b[1]-a[1])*t))
                    draw.line([a,end],fill=255,width=3)
            reveal = np.array(line)>0
        if i >= 35:
            bottom = 480 + round(65*min(1,(i-35)/18))
            reveal[480:bottom,853:919] = True
        mask = shape & reveal
        frame[mask] = target[mask]
        assert np.array_equal(frame[~shape],arr[~shape])
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait():
        raise RuntimeError('Video encoding failed')

if __name__ == '__main__':
    main()
