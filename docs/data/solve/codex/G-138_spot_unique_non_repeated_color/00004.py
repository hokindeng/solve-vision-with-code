from pathlib import Path
import subprocess
import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path('/app')

def main():
    base = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    # Isolate the sole purple shape and recover its six boundary corners.
    rgb = base.astype(np.int16)
    mask = ((rgb[:, :, 2] > rgb[:, :, 1] + 30) &
            (rgb[:, :, 0] < 180) & (rgb[:, :, 1] < 150)).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    corners = cv2.approxPolyDP(max(contours, key=cv2.contourArea), 1, True).reshape(-1, 2)
    # Start at the top and trace clockwise.
    corners = corners[::-1]
    corners = np.roll(corners, -np.argmin(corners[:, 1]), axis=0)
    path = np.vstack([corners, corners[0]]).astype(float)
    lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    proc = subprocess.Popen([
        'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
        '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16', '-i', '-',
        '-an', '-c:v', 'libx264', '-preset', 'slow', '-crf', '0',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out / 'video.mp4')
    ], stdin=subprocess.PIPE)
    for i in range(21):
        frame = base.copy()
        if i:
            remaining = lengths.sum() * i / 20
            points = [path[0]]
            for j, length in enumerate(lengths):
                if remaining >= length:
                    points.append(path[j + 1])
                    remaining -= length
                else:
                    points.append(path[j] + (path[j + 1] - path[j]) * remaining / length)
                    break
            scale = 4
            matte = Image.new('L', (1024 * scale, 1024 * scale))
            draw = ImageDraw.Draw(matte)
            coords = [tuple(p * scale) for p in points]
            draw.line(coords, fill=255, width=5 * scale, joint='curve')
            for x, y in coords:
                r = 2.5 * scale
                draw.ellipse((x-r, y-r, x+r, y+r), fill=255)
            alpha = np.array(matte.resize((1024, 1024), Image.Resampling.LANCZOS))
            affected = alpha > 0
            frame[affected] = np.rint(base[affected] * (1 - alpha[affected, None] / 255)).astype(np.uint8)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait():
        raise RuntimeError('Video encoding failed')

if __name__ == '__main__':
    main()
