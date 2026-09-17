from pathlib import Path
import subprocess
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    mask = np.all(original == (255, 165, 0), axis=2)
    yy, xx = np.where(mask)
    base = original.copy()
    base[mask] = (0, 255, 0)
    # Three cells left and four cells up: Manhattan distance seven.
    path = [(357,867),(255,867),(153,867),(51,867),
            (51,765),(51,663),(51,561),(51,459)]
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    proc = subprocess.Popen([
        'ffmpeg','-y','-loglevel','error','-f','rawvideo',
        '-vcodec','rawvideo','-pix_fmt','rgb24','-s','1024x1024',
        '-r','16','-i','-','-an','-c:v','libx264','-crf','0',
        '-preset','medium','-pix_fmt','yuv420p','-movflags','+faststart',
        str(out / 'video.mp4')], stdin=subprocess.PIPE)
    for frame in range(55):
        # Initial and final brief holds, with seven equally paced moves.
        progress = np.clip((frame - 2) / 49, 0, 1) * 7
        segment = min(int(progress), 6)
        fraction = progress - segment
        a, b = np.array(path[segment]), np.array(path[segment+1])
        center = np.rint(a + fraction * (b-a)).astype(int)
        canvas = base.copy()
        canvas[yy + center[1]-867, xx + center[0]-357] = (255,165,0)
        if frame == 0:
            canvas = original
        proc.stdin.write(canvas.tobytes())
    proc.stdin.close()
    if proc.wait():
        raise RuntimeError('ffmpeg encoding failed')

if __name__ == '__main__':
    main()
