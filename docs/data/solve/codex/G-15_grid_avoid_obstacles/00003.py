from pathlib import Path
from collections import deque
import numpy as np
from PIL import Image
import subprocess

ROOT = Path('/app')
def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    blue = np.array([0, 100, 255], dtype=np.uint8)
    # Extract the original agent, including its darker outline.
    mask = np.zeros(original.shape[:2], dtype=bool)
    mask[60:125, 340:405] = np.any(original[60:125, 340:405] != blue, axis=2)
    ys, xs = np.where(mask)
    x0, x1, y0, y1 = xs.min(), xs.max()+1, ys.min(), ys.max()+1
    sprite = original[y0:y1, x0:x1].copy()
    sprite_mask = mask[y0:y1, x0:x1]
    background = original.copy()
    background[mask] = blue

    start, goal = (3, 0), (8, 8)
    obstacles = {(7, 2), (5, 4), (1, 6), (9, 6), (3, 8), (4, 9)}
    queue = deque([start])
    parents = {start: None}
    while queue:
        cell = queue.popleft()
        if cell == goal:
            break
        for dx, dy in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
            nxt = (cell[0]+dx, cell[1]+dy)
            if 0 <= nxt[0] < 10 and 0 <= nxt[1] < 10 and nxt not in obstacles and nxt not in parents:
                parents[nxt] = cell
                queue.append(nxt)
    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = parents[node]
    path.reverse()
    assert len(path)-1 == 13
    (ROOT / 'output').mkdir(exist_ok=True)
    command = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-vcodec', 'rawvideo',
               '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16', '-i', '-', '-an',
               '-c:v', 'libx264', '-preset', 'slow', '-crf', '0', '-pix_fmt', 'yuv420p',
               '-movflags', '+faststart', str(ROOT / 'output/video.mp4')]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE)
    for frame_index in range(62):
        progress = min(frame_index / 59, 1) * (len(path)-1)
        segment = min(int(progress), len(path)-2)
        fraction = progress-segment
        a, b = path[segment], path[segment+1]
        cell_x = a[0] + fraction*(b[0]-a[0])
        cell_y = a[1] + fraction*(b[1]-a[1])
        px = x0 + round((cell_x-start[0])*93)
        py = y0 + round((cell_y-start[1])*93)
        frame = background.copy()
        target = frame[py:py+sprite.shape[0], px:px+sprite.shape[1]]
        target[sprite_mask] = sprite[sprite_mask]
        if frame_index == 0:
            assert np.array_equal(frame, original)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError('ffmpeg failed')

if __name__ == '__main__':
    main()
