from pathlib import Path
import numpy as np
from PIL import Image
import subprocess

ROOT = Path(__file__).resolve().parent

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    # Select just the red ink; the slot outline and label are unchanged.
    target = ((original[:,:,0] > 200) & (original[:,:,1] < 100)
              & (original[:,:,2] < 100))
    yy, xx = np.indices(target.shape)
    target &= (xx >= 680) & (xx <= 765) & (yy >= 465) & (yy <= 545)
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    encoder = subprocess.Popen([
        'ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', '1024x1024', '-r', '16', '-i', '-', '-an', '-c:v', 'libx264',
        '-crf', '0', '-preset', 'slow', '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart', str(out / 'video.mp4')
    ], stdin=subprocess.PIPE)
    for index in range(45):
        frame = original.copy()
        # A short initial hold, smooth fade across the duration, and final hold.
        t = np.clip((index - 3) / 37.0, 0.0, 1.0)
        fade = t * t * (3 - 2 * t)
        frame[target] = np.rint(original[target].astype(float) * (1-fade) + 255*fade).astype(np.uint8)
        # Deletion at the final position leaves no subsequent item to shift.
        encoder.stdin.write(frame.tobytes())
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError('Video encoding failed')

if __name__ == '__main__':
    main()
