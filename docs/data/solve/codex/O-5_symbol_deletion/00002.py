from pathlib import Path
import subprocess
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    red = (original[:,:,0] > 230) & (original[:,:,1] < 40) & (original[:,:,2] < 40)
    yy, xx = np.where(red)
    x0, x1 = int(xx.min()), int(xx.max()) + 1
    y0, y1 = int(yy.min()), int(yy.max()) + 1
    patch = original[y0:y1,x0:x1].astype(np.float64)
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    command = ['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','rgb24',
               '-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264',
               '-preset','slow','-crf','0','-pix_fmt','yuv420p',
               '-movflags','+faststart',str(out / 'video.mp4')]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE)
    for i in range(46):
        frame = original.copy()
        # Establish the marked target, then smoothly erase its symbol and border.
        progress = np.clip((i - 5) / 35, 0, 1)
        amount = progress * progress * (3 - 2 * progress)
        frame[y0:y1,x0:x1] = np.rint(patch * (1-amount) + 255 * amount).astype(np.uint8)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError('Video encoding failed')

if __name__ == '__main__':
    main()
