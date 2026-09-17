from pathlib import Path
from collections import deque
import subprocess
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
START = (4, 0, 2, 8, 1, 3, 5, 7, 6)
GOAL = (1, 2, 3, 4, 5, 6, 7, 8, 0)
COORD = (36, 364, 692)
SIZE = 296

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
            nxt = list(state)
            nxt[blank], nxt[source] = nxt[source], nxt[blank]
            nxt = tuple(nxt)
            if nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, moves + [state[source]]))

def xy(index):
    return COORD[index % 3], COORD[index // 3]

def main():
    original = Image.open(ROOT / 'first_frame.png').convert('RGB')
    background = original.copy()
    tiles = {}
    for index, number in enumerate(START):
        if number:
            x, y = xy(index)
            tiles[number] = original.crop((x, y, x + SIZE, y + SIZE))
            background.paste((255, 255, 255), (x, y, x + SIZE, y + SIZE))
    bg = np.array(background)
    grid_mask = np.any(bg != 255, axis=2)
    moves = solve()
    assert len(moves) == 11
    out = ROOT / 'output' / 'video.mp4'
    out.parent.mkdir(exist_ok=True)
    encoder = subprocess.Popen([
        'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
        '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16', '-i', '-',
        '-an', '-c:v', 'libx264', '-preset', 'slow', '-crf', '0',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out)
    ], stdin=subprocess.PIPE)
    def emit(frame):
        encoder.stdin.write(np.asarray(frame, dtype=np.uint8).tobytes())
    for _ in range(3):
        emit(original)
    state = list(START)
    for number in moves:
        source, target = state.index(number), state.index(0)
        sx, sy = xy(source)
        tx, ty = xy(target)
        for step in range(1, 7):
            frame = background.copy()
            for index, tile in enumerate(state):
                if tile and tile != number:
                    frame.paste(tiles[tile], xy(index))
            t = step / 6
            t = t * t * (3 - 2 * t)
            frame.paste(tiles[number], (round(sx + (tx - sx) * t), round(sy + (ty - sy) * t)))
            pixels = np.array(frame)
            pixels[grid_mask] = bg[grid_mask]
            emit(pixels)
        state[target], state[source] = state[source], state[target]
    assert tuple(state) == GOAL
    for _ in range(7):
        emit(pixels)
    encoder.stdin.close()
    if encoder.wait():
        raise RuntimeError('ffmpeg failed')
    print(f'{out}: 76 frames; moves {moves}')

if __name__ == '__main__':
    main()
