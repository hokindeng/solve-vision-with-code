from pathlib import Path
from collections import deque
import subprocess
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output' / 'video.mp4'
START = (2, 5, 3, 4, 6, 0, 7, 1, 8)
GOAL = (1, 2, 3, 4, 5, 6, 7, 8, 0)

def solve():
    queue = deque([(START, [])])
    seen = {START}
    while queue:
        state, moves = queue.popleft()
        if state == GOAL:
            return moves
        blank = state.index(0)
        for source in range(9):
            if abs(source // 3 - blank // 3) + abs(source % 3 - blank % 3) != 1:
                continue
            next_state = list(state)
            next_state[blank], next_state[source] = next_state[source], 0
            next_state = tuple(next_state)
            if next_state not in seen:
                seen.add(next_state)
                queue.append((next_state, moves + [state[source]]))
    raise RuntimeError('No solution')

def main():
    original = Image.open(ROOT / 'first_frame.png').convert('RGB')
    positions = [(36 + 328 * (i % 3), 36 + 328 * (i // 3)) for i in range(9)]
    background = original.copy()
    tiles = {}
    for i, number in enumerate(START):
        if number:
            x, y = positions[i]
            tiles[number] = original.crop((x, y, x + 296, y + 296))
            background.paste((255, 255, 255), (x, y, x + 296, y + 296))
    fixed = np.array(background)
    grid_mask = np.any(fixed != 255, axis=2)
    moves = solve()
    assert len(moves) == 11
    OUT.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen([
        'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', '1024x1024', '-r', '16', '-i', '-', '-an', '-c:v', 'libx264',
        '-crf', '0', '-preset', 'medium', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(OUT)
    ], stdin=subprocess.PIPE)
    count = 0
    def emit(frame):
        nonlocal count
        process.stdin.write(np.asarray(frame, dtype=np.uint8).tobytes())
        count += 1
    emit(original)
    state = list(START)
    for number in moves:
        source, target = state.index(number), state.index(0)
        assert abs(source // 3 - target // 3) + abs(source % 3 - target % 3) == 1
        sx, sy = positions[source]
        tx, ty = positions[target]
        for step in range(1, 7):
            t = step / 6
            t = t * t * (3 - 2 * t)
            frame = background.copy()
            for i, tile in enumerate(state):
                if tile and tile != number:
                    frame.paste(tiles[tile], positions[i])
            frame.paste(tiles[number], (round(sx + (tx - sx) * t), round(sy + (ty - sy) * t)))
            pixels = np.array(frame)
            pixels[grid_mask] = fixed[grid_mask]
            emit(pixels)
        state[target], state[source] = number, 0
    assert tuple(state) == GOAL
    for _ in range(9):
        emit(pixels)
    assert count == 76
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError('Video encoding failed')
    print(f'{OUT}: {count} frames, 11 legal moves, solved')

if __name__ == '__main__':
    main()
