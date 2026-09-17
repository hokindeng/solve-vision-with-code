from pathlib import Path
from collections import deque
import subprocess
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
START = (8, 0, 2, 1, 4, 3, 7, 6, 5)
GOAL = (1, 2, 3, 4, 5, 6, 7, 8, 0)

def solution():
    queue = deque([(START, [])])
    seen = {START}
    while queue:
        state, moves = queue.popleft()
        if state == GOAL:
            return moves
        blank = state.index(0)
        for pos in range(9):
            if abs(pos // 3 - blank // 3) + abs(pos % 3 - blank % 3) != 1:
                continue
            nxt = list(state)
            nxt[blank], nxt[pos] = nxt[pos], nxt[blank]
            nxt = tuple(nxt)
            if nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, moves + [state[pos]]))
    raise RuntimeError('No solution')

def main():
    original = Image.open(ROOT / 'first_frame.png').convert('RGB')
    positions = [(36 + 328 * (i % 3), 36 + 328 * (i // 3)) for i in range(9)]
    background = original.copy()
    sprites = {}
    for i, number in enumerate(START):
        if number:
            x, y = positions[i]
            sprites[number] = original.crop((x, y, x + 296, y + 296))
            background.paste((255, 255, 255), (x, y, x + 296, y + 296))
    base = np.array(background)
    grid = np.any(base != 255, axis=2)
    def render(state, moving=None, location=None):
        frame = background.copy()
        for i, number in enumerate(state):
            if number and number != moving:
                frame.paste(sprites[number], positions[i])
        if moving is not None:
            frame.paste(sprites[moving], location)
        pixels = np.array(frame)
        pixels[grid] = base[grid]
        return pixels

    moves = solution()
    assert len(moves) == 11
    out = ROOT / 'output' / 'video.mp4'
    out.parent.mkdir(exist_ok=True)
    encoder = subprocess.Popen([
        'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
        '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16', '-i', '-',
        '-an', '-c:v', 'libx264', '-preset', 'slow', '-crf', '18',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out)
    ], stdin=subprocess.PIPE)
    def write(frame):
        encoder.stdin.write(frame.tobytes())
    for _ in range(5):
        write(np.array(original))
    state = list(START)
    for number in moves:
        source, target = state.index(number), state.index(0)
        assert abs(source // 3 - target // 3) + abs(source % 3 - target % 3) == 1
        sx, sy = positions[source]
        tx, ty = positions[target]
        for step in range(1, 7):
            t = step / 6
            t = t * t * (3 - 2 * t)
            location = (round(sx + (tx - sx) * t), round(sy + (ty - sy) * t))
            write(render(state, number, location))
        state[source], state[target] = 0, number
    assert tuple(state) == GOAL
    for _ in range(5):
        write(render(state))
    encoder.stdin.close()
    assert encoder.wait() == 0
    print(f'Created {out}: 76 frames, 11 moves: {moves}')

if __name__ == '__main__':
    main()
