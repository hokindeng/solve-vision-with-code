from pathlib import Path
import numpy as np
from PIL import Image
import subprocess

ROOT = Path(__file__).resolve().parent

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    blue = np.array([37, 99, 235], dtype=np.uint8)
    # Missing cells, in top-to-bottom order (zero-based row and column).
    missing = [(1, 4), (1, 5), (4, 5), (5, 4)]
    edges = [204, 307, 409, 512, 614, 716, 820]
    masks = []
    for row, col in missing:
        mask = np.zeros(original.shape[:2], dtype=bool)
        mask[edges[row]:edges[row+1], edges[col]:edges[col+1]] = True
        mask &= np.all(original == 255, axis=2)
        masks.append(mask)
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
           '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024',
           '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '10',
           '-preset', 'slow', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
           str(out / 'video.mp4')]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for frame_number in range(35):
        frame = original.copy()
        for i, mask in enumerate(masks):
            t = np.clip((frame_number - i * 8) / 8, 0, 1)
            t = t*t*(3-2*t)
            frame[mask] = np.rint(255*(1-t) + blue.astype(float)*t).astype(np.uint8)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError('ffmpeg failed')

if __name__ == '__main__':
    main()
