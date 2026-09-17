from pathlib import Path
import subprocess
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    purple = np.array([124, 58, 237], dtype=np.uint8)
    # Cell interiors exclude every existing grid-line pixel.
    spans = [(206,306),(309,408),(411,511),(514,613),(616,715),(718,818)]
    missing = []
    for row, (y0,y1) in enumerate(spans):
        for col in range(3,6):
            x0,x1 = spans[col]
            left0,left1 = spans[5-col]
            if np.array_equal(original[(y0+y1)//2,(left0+left1)//2],purple) and not np.array_equal(original[(y0+y1)//2,(x0+x1)//2],purple):
                missing.append((y0,y1,x0,x1))
    assert len(missing) == 4
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    command = ['ffmpeg','-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-preset','slow','-crf','0','-pix_fmt','yuv420p','-movflags','+faststart',str(out/'video.mp4')]
    proc = subprocess.Popen(command,stdin=subprocess.PIPE)
    for frame_index in range(35):
        frame = original.copy()
        for i,(y0,y1,x0,x1) in enumerate(missing):
            progress = np.clip(frame_index / 34 * len(missing) - i, 0, 1)
            progress = progress * progress * (3 - 2 * progress)
            if progress > 0:
                source = original[y0:y1,x0:x1]
                frame[y0:y1,x0:x1] = np.rint(source.astype(float)*(1-progress)+purple*progress).astype(np.uint8)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait():
        raise RuntimeError('Video encoding failed')

if __name__ == '__main__':
    main()
