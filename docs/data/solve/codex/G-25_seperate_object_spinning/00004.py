from pathlib import Path
import cv2
import numpy as np
import subprocess

ROOT = Path('/app')
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)

def main():
    original = cv2.imread(str(ROOT / 'first_frame.png'))
    height, width = original.shape[:2]
    yy, xx = np.indices((height, width))
    moving_mask = (np.any(original != 255, axis=2) & (xx < 450)).astype(np.uint8)
    count, labels, stats, centers = cv2.connectedComponentsWithStats(moving_mask)
    # Image-coordinate angles: positive rotation is clockwise.
    motions = [(-17.0436365, 588.88365012),
               (23.89198139, 539.11997704),
               (0.0, 466.0),
               (8.01918734, 476.08663682),
               (-19.90630855, 454.69919322)]
    objects = []
    for index, (angle, distance) in enumerate(motions, 1):
        x, y, w, h, _ = stats[index]
        region = original[y:y+h, x:x+w]
        ink = (255 - region).astype(np.float32)
        ink[labels[y:y+h, x:x+w] != index] = 0
        objects.append((ink, np.array([x, y]), centers[index], angle, distance))
    background = original.copy()
    background[moving_mask != 0] = 255
    fixed = np.any(background != 255, axis=2)
    command = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
               '-vcodec', 'rawvideo', '-pix_fmt', 'bgr24', '-s', '1024x1024',
               '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '15',
               '-preset', 'medium', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
               str(OUT / 'video.mp4')]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    def ease(t):
        t = np.clip(t, 0, 1)
        return t*t*(3-2*t)
    for frame_index in range(48):
        if frame_index == 0:
            frame = original.copy()
        else:
            t = frame_index / 47
            rotation = ease(t / 0.42)
            translation = ease((t - 0.42) / 0.58)
            canvas = np.full(original.shape, 255, dtype=np.float32)
            for ink, origin, center, angle, distance in objects:
                radians = np.deg2rad(angle * rotation)
                matrix = np.array([[np.cos(radians), -np.sin(radians)],
                                   [np.sin(radians), np.cos(radians)]])
                offset = center + matrix @ (origin - center) + [distance * translation, 0]
                transform = np.column_stack((matrix, offset))
                warped = cv2.warpAffine(ink, transform, (width, height),
                                        flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
                canvas -= warped
            frame = np.clip(np.rint(canvas), 0, 255).astype(np.uint8)
            # Preserve every original dashed-outline pixel throughout the animation.
            frame[fixed] = background[fixed]
        encoder.stdin.write(frame.tobytes())
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError('Video encoding failed')

if __name__ == '__main__':
    main()
