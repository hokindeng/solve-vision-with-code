from pathlib import Path
import cv2
import numpy as np
import subprocess

ROOT = Path('/app')

def main():
    original = cv2.imread(str(ROOT / 'first_frame.png'))
    mask = np.any(original != 255, axis=2).astype(np.uint8)
    n, labels, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
    circles = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 100:
            continue
        circles.append(dict(x=int(x), y=int(y), w=int(w), h=int(h),
                            sprite=original[y:y+h, x:x+w].copy(),
                            mask=labels[y:y+h, x:x+w] == i))
    circles.sort(key=lambda c: c['w'], reverse=True)
    gap = 18
    total = sum(c['w'] for c in circles) + gap * (len(circles)-1)
    left = round((1024-total)/2)
    for c in circles:
        c['target'] = (left, 512 - c['h']//2)
        left += c['w'] + gap
    # Curved routes keep the small circles clear of the larger moving circles.
    arcs = {71: -280, 61: 180, 85: 110}
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
           '-vcodec', 'rawvideo', '-pix_fmt', 'bgr24', '-s', '1024x1024',
           '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '15',
           '-preset', 'medium', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
           str(out / 'video.mp4')]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for frame in range(80):
        t = frame / 79
        u = t*t*(3-2*t)
        canvas = np.full_like(original, 255)
        for c in circles:
            tx, ty = c['target']
            x = round(c['x'] + (tx-c['x'])*u)
            y = round(c['y'] + (ty-c['y'])*u + arcs.get(c['w'], 0)*np.sin(np.pi*u))
            region = canvas[y:y+c['h'], x:x+c['w']]
            region[c['mask']] = c['sprite'][c['mask']]
        if frame == 0:
            assert np.array_equal(canvas, original)
        proc.stdin.write(canvas.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError('ffmpeg encoding failed')

if __name__ == '__main__':
    main()
