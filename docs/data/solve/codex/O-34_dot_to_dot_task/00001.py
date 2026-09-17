from pathlib import Path
import cv2
import numpy as np
import subprocess

ROOT = Path('/app')
OUT = ROOT / 'output'
POINTS = [(557, 253), (147, 871), (595, 688), (357, 403), (768, 282)]

def main():
    OUT.mkdir(exist_ok=True)
    original = cv2.imread(str(ROOT / 'first_frame.png'))
    # Keep the numbered dots completely intact, with connections behind them.
    protected = np.zeros(original.shape[:2], dtype=np.uint8)
    for point in POINTS:
        cv2.circle(protected, point, 46, 255, -1)
    command = ['ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
               '-pix_fmt', 'bgr24', '-s', '1024x1024', '-r', '16',
               '-i', '-', '-an', '-c:v', 'libx264', '-crf', '18',
               '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
               str(OUT / 'video.mp4')]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    for frame_index in range(70):
        frame = original.copy()
        progress = 4 * frame_index / 69
        for segment in range(4):
            amount = min(1.0, max(0.0, progress - segment))
            if amount <= 0:
                continue
            a = np.array(POINTS[segment], dtype=float)
            b = np.array(POINTS[segment + 1], dtype=float)
            direction = (b - a) / np.linalg.norm(b - a)
            start = a + direction * 44
            end = b - direction * 44
            tip = start + amount * (end - start)
            cv2.line(frame, tuple(np.rint(start).astype(int)),
                     tuple(np.rint(tip).astype(int)), (0, 0, 255), 5, cv2.LINE_AA)
        frame[protected > 0] = original[protected > 0]
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    errors = process.stderr.read()
    if process.wait() != 0:
        raise RuntimeError(errors.decode())

if __name__ == '__main__':
    main()
