from pathlib import Path
import subprocess
import numpy as np
from PIL import Image

ROOT = Path('/app')

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    green = (original[:,:,1] > 240) & (original[:,:,0] < 10) & (original[:,:,2] < 10)
    ys, xs = np.where(green)
    cx, cy = (xs.min()+xs.max())/2, (ys.min()+ys.max())/2
    background = original.copy()
    background[green] = original[620,110]
    # Six cells right and two cells down is a shortest Manhattan path.
    points = [(cx + 102*i, cy) for i in range(7)]
    points += [(cx + 612, cy + 102*i) for i in (1,2)]
    output = ROOT / 'output'
    output.mkdir(exist_ok=True)
    process = subprocess.Popen([
        'ffmpeg','-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24',
        '-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264',
        '-crf','18','-preset','slow','-pix_fmt','yuv420p','-movflags','+faststart',
        str(output / 'video.mp4')], stdin=subprocess.PIPE)
    for frame in range(60):
        progress = np.clip((frame-2)/54,0,1)*8
        segment = min(int(progress),7)
        fraction = progress-segment
        x = points[segment][0]*(1-fraction)+points[segment+1][0]*fraction
        y = points[segment][1]*(1-fraction)+points[segment+1][1]*fraction
        canvas = background.copy()
        canvas[ys+int(round(y-cy)),xs+int(round(x-cx))] = original[ys,xs]
        if frame == 0:
            canvas = original
        process.stdin.write(canvas.tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError('ffmpeg failed')

if __name__ == '__main__':
    main()
