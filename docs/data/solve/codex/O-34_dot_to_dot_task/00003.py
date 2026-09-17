from pathlib import Path
import subprocess
import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    height, width = original.shape[:2]
    points = [(256, 721), (644, 195), (168, 136), (880, 439)]
    # Render connections beneath the original numbered circles.
    protected = np.zeros((height, width), np.uint8)
    for point in points:
        cv2.circle(protected, point, 46, 255, -1)
    output = ROOT / 'output' / 'video.mp4'
    output.parent.mkdir(parents=True, exist_ok=True)
    command = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
               '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{width}x{height}',
               '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '15',
               '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    for frame_number in range(55):
        frame = original.copy()
        for index in range(3):
            progress = np.clip((frame_number - index * 18) / 18, 0, 1)
            if progress <= 0:
                continue
            start = points[index]
            end = tuple(int(round(a + (b - a) * progress))
                        for a, b in zip(start, points[index + 1]))
            cv2.line(frame, start, end, (255, 0, 0), 5, cv2.LINE_AA)
        frame[protected > 0] = original[protected > 0]
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError('Video encoding failed')

if __name__ == '__main__':
    main()
