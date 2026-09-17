from pathlib import Path
from collections import deque
import subprocess
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
START = (2, 3, 6, 1, 5, 0, 7, 8, 4)
GOAL = (1, 2, 3, 4, 5, 6, 7, 8, 0)

def solve():
    queue = deque([(START, [])])
    seen = {START}
    while queue:
        state, path = queue.popleft()
        if state == GOAL:
            return path
        blank = state.index(0)
        for src in range(9):
            if abs(src // 3 - blank // 3) + abs(src % 3 - blank % 3) != 1:
                continue
            nxt = list(state)
            nxt[blank], nxt[src] = nxt[src], nxt[blank]
            nxt = tuple(nxt)
            if nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, path + [state[src]]))
    raise RuntimeError('No solution')

def main():
    original = Image.open(ROOT / 'first_frame.png').convert('RGB')
    coords = [(36 + 328 * (i % 3), 36 + 328 * (i // 3)) for i in range(9)]
    base = original.copy()
    tiles = {}
    for i, number in enumerate(START):
        x, y = coords[i]
        if number:
            tiles[number] = original.crop((x, y, x + 296, y + 296))
            base.paste((255, 255, 255), (x, y, x + 296, y + 296))
    # Restore fixed grid lines above each moving tile throughout its transit.
    fixed = np.zeros((1024, 1024), dtype=bool)
    for p in (20, 348, 675, 1003):
        fixed[20:1005, p:p + 2] = True
        fixed[p:p + 2, 20:1005] = True
    source = np.array(original)

    def render(state, moving=None, destination=None, fraction=0):
        canvas = base.copy()
        for i, number in enumerate(state):
            if number and number != moving:
                canvas.paste(tiles[number], coords[i])
        if moving is not None:
            x, y = coords[state.index(moving)]
            tx, ty = coords[destination]
            canvas.paste(tiles[moving], (round(x + (tx-x)*fraction), round(y + (ty-y)*fraction)))
        frame = np.array(canvas)
        frame[fixed] = source[fixed]
        return frame

    moves = solve()
    assert len(moves) == 11
    assert np.array_equal(render(START), source)
    output = ROOT / 'output'
    output.mkdir(exist_ok=True)
    proc = subprocess.Popen([
        'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
        '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024',
        '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'slow',
        '-crf', '0', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
        str(output / 'video.mp4')], stdin=subprocess.PIPE)
    count = 0
    def emit(frame):
        nonlocal count
        proc.stdin.write(frame.tobytes())
        count += 1
    for _ in range(4):
        emit(source)
    state = list(START)
    for number in moves:
        src, blank = state.index(number), state.index(0)
        assert abs(src//3 - blank//3) + abs(src%3 - blank%3) == 1
        for step in range(1, 7):
            t = step / 6
            t = t*t*(3-2*t)
            emit(render(state, number, blank, t))
        state[src], state[blank] = state[blank], state[src]
    assert tuple(state) == GOAL
    for _ in range(6):
        emit(render(state))
    proc.stdin.close()
    assert proc.wait() == 0
    assert count == 76
    print('Created', output / 'video.mp4', 'with', count, 'frames. Moves:', moves)

if __name__ == '__main__':
    main()
