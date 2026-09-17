from pathlib import Path
import subprocess
import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output' / 'video.mp4'

def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    color = np.array([20, 184, 166], dtype=np.uint8)
    pipe_pixels = np.all(original == color, axis=2)
    background = original.copy()
    background[pipe_pixels] = [255, 255, 255]
    # Tile-center pivots remain fixed. In screen coordinates positive angles
    # turn clockwise. Final elbows are SE, SW, NE, NW respectively.
    tiles = [(387, 387, 360.0), (637, 387, 66.0),
             (387, 637, 36.0), (637, 637, 163.0)]
    masks = []
    for cx, cy, degrees in tiles:
        mask = np.zeros((1024, 1024), dtype=np.uint8)
        mask[cy-107:cy+108, cx-107:cx+108] = (
            pipe_pixels[cy-107:cy+108, cx-107:cx+108] * 255)
        masks.append(mask)
    proc = subprocess.Popen([
        'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
        '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024',
        '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'slow',
        '-crf', '15', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(OUT)
    ], stdin=subprocess.PIPE)
    try:
        for i in range(96):
            if i == 0:
                frame = original
            else:
                t = i / 95.0
                ease = t*t*t*(10.0 - 15.0*t + 6.0*t*t)
                frame = background.copy()
                for (cx, cy, degrees), mask in zip(tiles, masks):
                    transform = cv2.getRotationMatrix2D((cx, cy), -degrees*ease, 1)
                    rotated = cv2.warpAffine(mask, transform, (1024, 1024),
                                             flags=cv2.INTER_LINEAR)
                    active = rotated > 0
                    alpha = rotated[active, None].astype(np.float32) / 255
                    frame[active] = np.rint(frame[active]*(1-alpha) + color*alpha).astype(np.uint8)
            proc.stdin.write(frame.tobytes())
    finally:
        proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError('Video encoding failed')

if __name__ == '__main__':
    main()
