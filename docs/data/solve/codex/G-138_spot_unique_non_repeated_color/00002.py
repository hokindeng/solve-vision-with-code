from pathlib import Path
import subprocess
import cv2
import numpy as np
from PIL import Image

ROOT = Path('/app')

def main():
    base = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    mask = np.uint8(np.all(base == (201, 87, 236), axis=2)) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)[:, 0, :]
    start = np.lexsort((contour[:, 0], contour[:, 1]))[0]
    contour = np.roll(contour, -start, axis=0)
    # Travel clockwise from the top of the circle.
    if contour[1, 0] < contour[0, 0]:
        contour = np.concatenate((contour[:1], contour[:0:-1]))
    path = np.concatenate((contour, contour[:1])).astype(float)
    lengths = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))))
    (ROOT / 'output').mkdir(exist_ok=True)
    proc = subprocess.Popen([
        'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16', '-i', '-', '-an',
        '-c:v', 'libx264', '-crf', '10', '-preset', 'slow', '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart', str(ROOT / 'output/video.mp4')
    ], stdin=subprocess.PIPE)
    for i in range(21):
        frame = base.copy()
        if i:
            distance = lengths[-1] * i / 20
            j = min(np.searchsorted(lengths, distance, side='right') - 1, len(path)-2)
            endpoint = path[j] + (path[j+1]-path[j]) * ((distance-lengths[j])/(lengths[j+1]-lengths[j]))
            points = np.vstack((path[:j+1], endpoint))
            scale = 4
            ink = np.zeros((1024*scale, 1024*scale), np.uint8)
            cv2.polylines(ink, [np.rint(points*scale).astype(np.int32)], False, 255, 4*scale, cv2.LINE_AA)
            alpha = cv2.resize(ink, (1024,1024), interpolation=cv2.INTER_AREA).astype(float)/255
            frame = np.rint(base * (1-alpha[:,:,None])).astype(np.uint8)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait():
        raise RuntimeError('ffmpeg encoding failed')

if __name__ == '__main__':
    main()
