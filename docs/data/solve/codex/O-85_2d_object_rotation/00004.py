from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import subprocess

ROOT = Path(__file__).resolve().parent

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    h, w = original.shape[:2]
    # The only foreground below the heading is the two outlined objects.
    mask = np.any(original < 250, axis=2).astype(np.uint8)
    mask[:150] = 0
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    objects = []
    background = original.copy()
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] < 100:
            continue
        x, y, bw, bh, _ = stats[label]
        # Include the full antialiased edge in the isolated white patch.
        pad = 5
        x0, y0 = x-pad, y-pad
        x1, y1 = x+bw+pad, y+bh+pad
        layer = np.full_like(original, 255)
        layer[y0:y1, x0:x1] = original[y0:y1, x0:x1]
        background[y0:y1, x0:x1] = 255
        silhouette = (labels == label).astype(np.uint8)
        moments = cv2.moments(silhouette)
        center = (moments['m10']/moments['m00'], moments['m01']/moments['m00'])
        objects.append((layer, center))
    assert len(objects) == 2
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    command = ['ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
               '-pix_fmt', 'rgb24', '-s', f'{w}x{h}', '-r', '16', '-i', '-',
               '-an', '-c:v', 'libx264', '-preset', 'slow', '-crf', '0',
               '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out/'video.mp4')]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    for frame_index in range(17):
        if frame_index == 0:
            frame = original.copy()
        else:
            frame = background.copy()
            angle = 21 * frame_index / 16
            for layer, center in objects:
                transform = cv2.getRotationMatrix2D(center, -angle, 1.0)
                rotated = cv2.warpAffine(layer, transform, (w, h), flags=cv2.INTER_CUBIC,
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
                visible = np.any(rotated != 255, axis=2)
                frame[visible] = rotated[visible]
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    stderr = proc.stderr.read()
    if proc.wait() != 0:
        raise RuntimeError(stderr.decode())

if __name__ == '__main__':
    main()
