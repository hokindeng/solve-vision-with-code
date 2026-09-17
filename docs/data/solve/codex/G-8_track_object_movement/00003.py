from pathlib import Path
import subprocess
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    # This region contains just the moving octagon and its green outline.
    x0, y0, x1, y1 = 230, 444, 439, 654
    sprite = original[y0:y1, x0:x1].copy()
    mask = np.any(sprite != 255, axis=2)
    base = original.copy()
    base[y0:y1, x0:x1][mask] = 255
    # Measure the outlined object's center and the red star's center.
    green = (original[:,:,1] > 170) & (original[:,:,0] < 80) & (original[:,:,2] < 80)
    red = (original[:,:,0] > 200) & (original[:,:,1] < 60) & (original[:,:,2] < 60)
    gx = np.where(green)[1]
    rx = np.where(red)[1]
    displacement = int(round((rx.min() + rx.max()) / 2 - (gx.min() + gx.max()) / 2))
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    command = ['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
               '-s', '1024x1024', '-r', '16', '-i', '-', '-an', '-c:v', 'libx264',
               '-preset', 'slow', '-crf', '0', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
               str(out / 'video.mp4')]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    for i in range(60):
        t = i / 59
        shift = round(displacement * (3*t*t - 2*t*t*t))
        frame = base.copy()
        region = frame[y0:y1, x0+shift:x1+shift]
        region[mask] = sprite[mask]
        if i == 0:
            assert np.array_equal(frame, original)
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError('ffmpeg failed')

if __name__ == '__main__':
    main()
