from pathlib import Path
import cv2
import numpy as np
import subprocess

ROOT = Path(__file__).resolve().parent

def main():
    source = cv2.cvtColor(cv2.imread(str(ROOT / 'first_frame.png')), cv2.COLOR_BGR2RGB)
    height, width = source.shape[:2]
    colors = [np.array([60, 101, 53]), np.array([188, 74, 153])]
    foreground = np.any(source != 255, axis=2).astype(np.uint8)
    _, labels, _, centers = cv2.connectedComponentsWithStats(foreground)
    masks, interiors, positions = [], [], []
    for color in colors:
        interior = np.all(source == color, axis=2).astype(np.uint8)
        label = int(labels[interior.astype(bool)][0])
        masks.append((labels == label).astype(np.uint8))
        interiors.append(interior)
        positions.append(centers[label])
    midpoint = np.mean(positions, axis=0)
    background = source.copy()
    background[foreground.astype(bool)] = 255
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    command = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-vcodec', 'rawvideo',
               '-pix_fmt', 'rgb24', '-s', f'{width}x{height}', '-r', '16', '-i', '-',
               '-an', '-c:v', 'libx264', '-crf', '16', '-preset', 'medium',
               '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out / 'video.mp4')]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    for index in range(80):
        if index == 0:
            frame = source
        else:
            progress = index / 79.0
            union = np.zeros((height, width), dtype=bool)
            light = np.zeros((height, width, 3), dtype=np.uint16)
            for mask, interior, position, color in zip(masks, interiors, positions, colors):
                displacement = (midpoint - position) * progress
                matrix = np.float32([[1, 0, displacement[0]], [0, 1, displacement[1]]])
                moved_mask = cv2.warpAffine(mask, matrix, (width, height), flags=cv2.INTER_NEAREST)
                moved_fill = cv2.warpAffine(interior, matrix, (width, height), flags=cv2.INTER_NEAREST)
                union |= moved_mask.astype(bool)
                light += moved_fill[:, :, None].astype(np.uint16) * color.astype(np.uint16)
            frame = background.copy()
            frame[union] = np.minimum(light[union], 255).astype(np.uint8)
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError('ffmpeg encoding failed')

if __name__ == '__main__':
    main()
