from pathlib import Path
import cv2
import numpy as np
import subprocess

ROOT = Path(__file__).resolve().parent

def main():
    source = cv2.imread(str(ROOT / 'first_frame.png'))
    height, width = source.shape[:2]
    foreground = (source.min(axis=2) < 250).astype(np.uint8)
    count, labels, stats, centers = cv2.connectedComponentsWithStats(foreground, 8)
    objects = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] > 1000]
    background = source.copy()
    layers = []
    for index in objects:
        mask = labels == index
        layer = np.full_like(source, 255)
        layer[mask] = source[mask]
        background[mask] = 255
        layers.append((layer, tuple(centers[index])))
    output = ROOT / 'output'
    output.mkdir(exist_ok=True)
    command = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
               '-pix_fmt', 'bgr24', '-s', f'{width}x{height}', '-r', '16',
               '-i', '-', '-an', '-c:v', 'libx264', '-crf', '15',
               '-preset', 'slow', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
               str(output / 'video.mp4')]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    for frame_number in range(17):
        if frame_number == 0:
            frame = source.copy()
        else:
            frame = background.copy()
            angle = 64.0 * frame_number / 16
            for layer, center in layers:
                # OpenCV's positive angles rotate counterclockwise in image space.
                transform = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(layer, transform, (width, height),
                                         flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_CONSTANT,
                                         borderValue=(255, 255, 255))
                frame = np.minimum(frame, rotated)
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError('ffmpeg encoding failed')

if __name__ == '__main__':
    main()
