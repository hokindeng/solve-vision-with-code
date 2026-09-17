from pathlib import Path
import subprocess
import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent

def main():
    src = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    h, w = src.shape[:2]
    foreground = np.any(src < 245, axis=2).astype(np.uint8)
    foreground[:100] = 0
    count, labels, stats, centers = cv2.connectedComponentsWithStats(foreground, 8)
    ids = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] > 500]
    assert len(ids) == 4
    background = src.copy()
    layers = []
    for i in ids:
        mask = labels == i
        # Preserve each original silhouette, including its black outline.
        layer = np.full_like(src, 255)
        layer[mask] = src[mask]
        background[mask] = 255
        layers.append((layer, tuple(centers[i])))
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
           '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}',
           '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '15',
           '-preset', 'slow', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
           str(out / 'video.mp4')]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for frame_index in range(17):
        if frame_index == 0:
            frame = src
        else:
            angle = 163.0 * frame_index / 16
            frame = background.copy()
            for layer, center in layers:
                transform = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(layer, transform, (w, h),
                                         flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_CONSTANT,
                                         borderValue=(255, 255, 255))
                frame = np.minimum(frame, rotated)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError('ffmpeg failed')

if __name__ == '__main__':
    main()
