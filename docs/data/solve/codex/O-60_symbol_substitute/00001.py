from pathlib import Path
import subprocess
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent

def smooth(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)

def main():
    base = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    yy, xx = np.indices(base.shape[:2])
    old = (xx >= 310) & (xx <= 399) & (yy >= 470) & (yy <= 553) & (base[:,:,0] == 255) & (base[:,:,1] == 255) & (base[:,:,2] == 0)
    # Copy the reference's actual raster outline, translating its design center
    # (947, 77) to the center of the third symbol (354, 512).
    reference = (xx > 900) & (xx < 990) & (yy > 30) & (yy < 125) & (base[:,:,0] == 238) & (base[:,:,1] == 130) & (base[:,:,2] == 238)
    sy, sx = np.where(reference)
    ty, tx = sy + 435, sx - 593
    star_colors = base[sy, sx].astype(float)
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    cmd = ['ffmpeg','-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-preset','slow','-crf','0','-pix_fmt','yuv420p','-movflags','+faststart',str(out/'video.mp4')]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(52):
        t = i / 51
        fade_out = smooth(t / 0.48)
        fade_in = smooth((t - 0.52) / 0.48)
        frame = base.copy()
        frame[old] = np.rint(base[old].astype(float) * (1-fade_out) + 255 * fade_out).astype(np.uint8)
        if fade_in > 0:
            frame[ty, tx] = np.rint(255 * (1-fade_in) + star_colors * fade_in).astype(np.uint8)
        p.stdin.write(frame.tobytes())
    p.stdin.close()
    if p.wait():
        raise RuntimeError('ffmpeg failed')

if __name__ == '__main__':
    main()
