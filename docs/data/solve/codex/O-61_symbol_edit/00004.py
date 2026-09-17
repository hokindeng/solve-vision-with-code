from pathlib import Path
import numpy as np
from PIL import Image
import subprocess

ROOT = Path('/app')

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    # Isolate only the five sequence symbols. All panel lines and labels are
    # retained directly from the supplied frame.
    sprites = []
    base = original.copy()
    for i in range(5):
        x = 44 + 105*i
        patch = original[465:561, x+1:x+97].copy()
        mask = (patch.max(axis=2).astype(int)-patch.min(axis=2).astype(int)) > 5
        yy, xx = np.where(mask)
        sprites.append((xx+x+1, yy+465, patch[mask]))
        base[yy+465, xx+x+1] = 255

    allowed = np.zeros((1024,1024), bool)
    for i in range(9):
        x = 44 + 105*i
        allowed[465:561, x+1:x+97] = True

    def ease(t):
        t = np.clip(t, 0, 1)
        return t*t*(3-2*t)

    def draw(frame, sprite, dx, opacity=1):
        xx, yy, colors = sprite
        xx = xx + int(round(dx))
        ok = allowed[yy,xx]
        xx, yy, colors = xx[ok], yy[ok], colors[ok]
        frame[yy,xx] = np.rint(colors*opacity + frame[yy,xx]*(1-opacity)).astype(np.uint8)

    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    writer = subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-crf','0','-preset','slow','-pix_fmt','yuv420p',str(out/'video.mp4')], stdin=subprocess.PIPE)
    for f in range(54):
        if f == 0:
            frame = original.copy()
        else:
            frame = base.copy()
            shift1 = ease((f-5)/14)
            shift2 = ease((f-29)/14)
            for i, sprite in enumerate(sprites):
                dx = 105*shift1 if i >= 2 else 0
                if i >= 3:
                    dx += 105*shift2
                draw(frame, sprite, dx)
            if f >= 20:
                draw(frame, sprites[3], -105, ease((f-19)/7))
            if f >= 44:
                draw(frame, sprites[3], 105, ease((f-43)/7))
        writer.stdin.write(frame.tobytes())
    writer.stdin.close()
    if writer.wait() != 0:
        raise RuntimeError('Video encoding failed')

if __name__ == '__main__':
    main()
