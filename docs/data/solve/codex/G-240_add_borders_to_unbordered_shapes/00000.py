from pathlib import Path
import cv2
import numpy as np
import subprocess
from PIL import Image

ROOT = Path(__file__).resolve().parent

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    height, width = original.shape[:2]
    yy, xx = np.indices((height, width))
    masks = [
        np.all(original == (100, 140, 220), axis=2),
        np.all(original == (255, 215, 0), axis=2),
        np.all(original == (255, 182, 193), axis=2) & (xx > 500),
    ]
    paths = []
    for mask in masks:
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        points = max(contours, key=cv2.contourArea)[:, 0, :]
        # Begin each stroke at its uppermost point and trace clockwise.
        points = points[::-1]
        start = np.lexsort((points[:, 0], points[:, 1]))[0]
        points = np.concatenate((points[start:], points[:start], points[start:start+1]))
        paths.append(points)
    output = ROOT / 'output'
    output.mkdir(exist_ok=True)
    process = subprocess.Popen([
        'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', f'{width}x{height}', '-r', '16', '-i', '-', '-an',
        '-c:v', 'libx264', '-preset', 'slow', '-crf', '0', '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart', str(output / 'video.mp4')
    ], stdin=subprocess.PIPE)
    for frame_index in range(80):
        frame = original.copy()
        for shape_index, (mask, points) in enumerate(zip(masks, paths)):
            progress = np.clip((frame_index - shape_index * 26) / 26.0, 0, 1)
            if progress <= 0:
                continue
            count = max(2, int(progress * (len(points) - 1)) + 1)
            stroke = np.zeros((height, width), np.uint8)
            cv2.polylines(stroke, [points[:count]], False, 255, 6, cv2.LINE_8)
            # Place the border inside the silhouette, preserving every background pixel.
            frame[(stroke > 0) & mask] = 0
        if frame_index == 0:
            assert np.array_equal(frame, original)
        assert np.array_equal(frame[~np.logical_or.reduce(masks)], original[~np.logical_or.reduce(masks)])
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError('Video encoding failed')

if __name__ == '__main__':
    main()
