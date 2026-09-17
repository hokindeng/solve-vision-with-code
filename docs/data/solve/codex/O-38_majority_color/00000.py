from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import subprocess

ROOT = Path('/app')

def main():
    src = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    # Include the black outlines in each object's mask.
    foreground = np.any(src < 250, axis=2).astype(np.uint8)
    n, labels = cv2.connectedComponents(foreground, connectivity=8)
    remove = np.zeros(src.shape[:2], bool)
    counts = {}
    components = []
    for i in range(1, n):
        mask = labels == i
        pixels = src[mask].astype(int)
        saturated = pixels.max(axis=1) - pixels.min(axis=1) > 100
        if not saturated.any():
            continue
        color = int(np.argmax(pixels[saturated].mean(axis=0)))
        components.append((mask, color))
        # Count colored interiors separately, including touching objects.
    for color, name in enumerate(('red', 'green', 'blue')):
        p = src.astype(int)
        interior = ((p[:,:,color] > 150) & (p[:,:,color] - np.max(np.delete(p,color,axis=2),axis=2) > 100)).astype(np.uint8)
        counts[name] = cv2.connectedComponents(interior, connectivity=8)[0] - 1
    majority = max(range(3), key=lambda c: counts[('red','green','blue')[c]])
    for mask, color in components:
        if color != majority:
            remove |= mask
    print('Object counts:', counts, '; majority:', ('red','green','blue')[majority])
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    cmd = ['ffmpeg','-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-crf','0','-preset','medium','-pix_fmt','yuv420p',str(out/'video.mp4')]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for t in range(40):
        frame = src.copy()
        u = t / 39
        fade = u*u*(3-2*u)
        frame[remove] = np.rint(src[remove].astype(float)*(1-fade)+255*fade).astype(np.uint8)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError('ffmpeg failed')

if __name__ == '__main__':
    main()
