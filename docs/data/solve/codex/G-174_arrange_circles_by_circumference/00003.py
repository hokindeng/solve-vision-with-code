from pathlib import Path
import cv2
import numpy as np
import subprocess

ROOT = Path(__file__).resolve().parent

def main():
    source = cv2.imread(str(ROOT / 'first_frame.png'))
    height, width = source.shape[:2]
    count, labels, stats, centers = cv2.connectedComponentsWithStats(
        np.any(source != 255, axis=2).astype(np.uint8), 8)
    objects = []
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        objects.append(dict(x=int(x), y=int(y), w=int(w), h=int(h),
                            pixels=source[y:y+h, x:x+w].copy(),
                            mask=labels[y:y+h, x:x+w] == label))
    objects.sort(key=lambda obj: obj['w'], reverse=True)
    gap = 22
    row_width = sum(obj['w'] for obj in objects) + gap * (len(objects)-1)
    cursor = (width-row_width)//2
    for obj in objects:
        obj['end_x'] = cursor
        obj['end_y'] = 512 - obj['h']//2
        cursor += obj['w'] + gap
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    command = ['ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
               '-pix_fmt', 'bgr24', '-s', f'{width}x{height}', '-r', '16',
               '-i', '-', '-an', '-c:v', 'libx264', '-crf', '18',
               '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out/'video.mp4')]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    for frame_number in range(80):
        t = frame_number / 79
        ease = t*t*t*(10 + t*(-15 + 6*t))
        frame = np.full_like(source, 255)
        for obj in objects:
            x = round(obj['x'] + (obj['end_x']-obj['x'])*ease)
            y = round(obj['y'] + (obj['end_y']-obj['y'])*ease)
            region = frame[y:y+obj['h'], x:x+obj['w']]
            region[obj['mask']] = obj['pixels'][obj['mask']]
        if frame_number == 0:
            assert np.array_equal(frame, source)
        encoder.stdin.write(frame.tobytes())
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError('ffmpeg encoding failed')

if __name__ == '__main__':
    main()
