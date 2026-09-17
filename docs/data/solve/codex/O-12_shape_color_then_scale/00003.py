from pathlib import Path
import subprocess
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
BASE = Image.open(ROOT / 'first_frame.png').convert('RGB')
# Preserve the source octagon's exact raster outline and change only its fill.
shape = np.array(BASE.crop((110, 612, 251, 754)))
purple = np.all(shape == (132, 30, 153), axis=2)
shape[purple] = (132, 153, 30)
OCTAGON = Image.fromarray(shape)

def ease(t):
    t = min(1., max(0., t))
    return t*t*(3-2*t)

def answer(im, cx, alpha, scale=1.):
    box = (cx-74, 608, cx+75, 758)
    before = BASE.crop(box)
    after = Image.new('RGB', before.size, 'white')
    w, h = round(OCTAGON.width*scale), round(OCTAGON.height*scale)
    octagon = OCTAGON if scale == 1 else OCTAGON.resize((w,h), Image.Resampling.LANCZOS)
    after.paste(octagon, (74-w//2, 75-h//2))
    im.paste(Image.blend(before, after, alpha), box[:2])

def frame(i):
    im = BASE.copy()
    # First answer resolves during the first half of the analogy.
    if i > 6:
        answer(im, 518, ease((i-6)/20))
    # The second answer appears at the large size, then contracts to medium.
    if i > 30:
        alpha = ease((i-30)/10)
        scale = 1 - (2/7)*ease((i-40)/16)
        answer(im, 854, alpha, scale)
    return np.asarray(im)

def main():
    (ROOT/'output').mkdir(exist_ok=True)
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
           '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024',
           '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'slow',
           '-crf', '0', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
           str(ROOT/'output/video.mp4')]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(60):
        p.stdin.write(frame(i).tobytes())
    p.stdin.close()
    if p.wait():
        raise RuntimeError('ffmpeg failed')
    # Validate that source pixels outside the two answer cells never change.
    mask = np.ones((1024,1024), dtype=bool)
    for cx in (518,854):
        mask[608:758,cx-74:cx+75] = False
    base = np.asarray(BASE)
    assert np.array_equal(frame(0), base)
    for i in range(60):
        assert np.array_equal(frame(i)[mask], base[mask])

if __name__ == '__main__':
    main()
