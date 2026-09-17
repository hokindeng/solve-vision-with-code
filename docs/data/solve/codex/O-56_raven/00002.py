from pathlib import Path
import numpy as np
from PIL import Image
import subprocess

ROOT = Path(__file__).resolve().parent

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    # Each row cyclically shifts the same three symbols: notch, square, diamond.
    # Copy the existing square's pixels to keep its exact color and stroke.
    square = original[110:231, 451:572].copy()
    square_mask = np.any(square < 250, axis=2)
    sy, sx = np.nonzero(square_mask)
    dx, dy = sx + 792, sy + 792
    colors = square[sy, sx]
    # Clockwise distance around the square, beginning at its top-left corner.
    x, y = sx - 3, sy - 4
    distances = np.stack([abs(y), abs(x-113), abs(y-112), abs(x)], axis=1)
    edge = distances.argmin(axis=1)
    progress = np.select([edge==0, edge==1, edge==2, edge==3],
                         [x, 113+y, 225+113-x, 338+112-y]) / 450
    question = np.zeros(original.shape[:2], dtype=bool)
    question[790:910, 810:890] = np.any(original[790:910, 810:890] != 255, axis=2)
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    proc = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16',
        '-i', '-', '-an', '-c:v', 'libx264', '-crf', '0', '-preset', 'medium',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out/'video.mp4')], stdin=subprocess.PIPE)
    for i in range(35):
        frame = original.copy()
        fade = min(i / 8, 1)
        frame[question] = np.rint(original[question]*(1-fade) + 255*fade).astype(np.uint8)
        if i >= 9:
            visible = progress <= min((i-8)/24, 1)
            frame[dy[visible], dx[visible]] = colors[visible]
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait():
        raise RuntimeError('ffmpeg failed')

if __name__ == '__main__':
    main()
