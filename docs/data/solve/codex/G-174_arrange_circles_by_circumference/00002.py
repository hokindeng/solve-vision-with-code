from pathlib import Path
import subprocess
import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent

def main():
    src = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    foreground = np.any(src != 255, axis=2).astype(np.uint8)
    count, labels, stats, centers = cv2.connectedComponentsWithStats(foreground, 8)
    circles = []
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        circles.append(dict(sprite=src[y:y+h, x:x+w].copy(),
                            mask=labels[y:y+h, x:x+w] == i,
                            start=np.array([x, y], dtype=float), w=int(w), h=int(h)))
    circles.sort(key=lambda c: c['w'], reverse=True)
    gap = 20
    width = sum(c['w'] for c in circles) + gap * (len(circles)-1)
    x = (1024-width)//2
    for c in circles:
        c['end'] = np.array([x, 512-c['h']//2], dtype=float)
        x += c['w']+gap
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    command = ['ffmpeg','-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo',
               '-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-',
               '-an','-c:v','libx264','-crf','15','-preset','medium',
               '-pix_fmt','yuv420p','-movflags','+faststart',str(out/'video.mp4')]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE)
    for frame in range(80):
        t = frame/79
        u = t*t*(3-2*t)
        canvas = np.full_like(src, 255)
        for c in circles:
            pos = (1-u)*c['start'] + u*c['end']
            # Lift the yellow circle as it passes the green circle.
            if c['w'] == 85:
                pos[1] -= 150*np.sin(np.pi*u)
            x,y = np.rint(pos).astype(int)
            region = canvas[y:y+c['h'],x:x+c['w']]
            region[c['mask']] = c['sprite'][c['mask']]
        if frame == 0:
            assert np.array_equal(canvas, src)
        proc.stdin.write(canvas.tobytes())
    proc.stdin.close()
    if proc.wait():
        raise RuntimeError('ffmpeg failed')

if __name__ == '__main__':
    main()
