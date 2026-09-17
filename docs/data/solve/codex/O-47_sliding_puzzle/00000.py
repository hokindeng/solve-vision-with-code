from pathlib import Path
from collections import deque
import numpy as np
from PIL import Image
import subprocess

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
original = Image.open(ROOT / 'first_frame.png').convert('RGB')
start = (5, 1, 3, 0, 7, 6, 2, 4, 8)
goal = (1, 2, 3, 4, 5, 6, 7, 8, 0)
# Find a shortest legal path.
queue = deque([start])
parents = {start: None}
while queue:
    state = queue.popleft()
    if state == goal:
        break
    blank = state.index(0)
    for source in range(9):
        if abs(source // 3 - blank // 3) + abs(source % 3 - blank % 3) != 1:
            continue
        nxt = list(state)
        nxt[blank], nxt[source] = nxt[source], nxt[blank]
        nxt = tuple(nxt)
        if nxt not in parents:
            parents[nxt] = state
            queue.append(nxt)
path = []
state = goal
while parents[state] is not None:
    prev = parents[state]
    path.append(state[prev.index(0)])
    state = prev
moves = path[::-1]
assert len(moves) == 11

positions = [(36 + 328 * (i % 3), 36 + 328 * (i // 3)) for i in range(9)]
size = 296
sprites = {}
background = original.copy()
for i, number in enumerate(start):
    x, y = positions[i]
    if number:
        sprites[number] = original.crop((x, y, x + size, y + size))
        background.paste((255, 255, 255), (x, y, x + size, y + size))
# The thin grid is static and remains visible during every slide.
bg_array = np.asarray(background)
grid = np.any(bg_array != 255, axis=2)

def draw(state, moving=None, destination=None, fraction=0):
    frame = background.copy()
    for i, number in enumerate(state):
        if not number:
            continue
        x, y = positions[i]
        if number == moving:
            dx, dy = positions[destination]
            x = round(x + (dx - x) * fraction)
            y = round(y + (dy - y) * fraction)
        frame.paste(sprites[number], (x, y))
    array = np.array(frame)
    array[grid] = bg_array[grid]
    return array

frames = [np.array(original)] * 4
state = list(start)
for number in moves:
    source, blank = state.index(number), state.index(0)
    assert abs(source // 3 - blank // 3) + abs(source % 3 - blank % 3) == 1
    for step in range(1, 7):
        t = step / 6
        t = t * t * (3 - 2 * t)
        frames.append(draw(state, number, blank, t))
    state[blank], state[source] = state[source], state[blank]
assert tuple(state) == goal
frames.extend([draw(state)] * 6)
assert len(frames) == 76
assert np.array_equal(frames[0], np.asarray(original))
encoder = subprocess.Popen([
    'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
    '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024',
    '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '15',
    '-preset', 'slow', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
    str(OUT / 'video.mp4')
], stdin=subprocess.PIPE)
for frame in frames:
    encoder.stdin.write(frame.tobytes())
encoder.stdin.close()
assert encoder.wait() == 0
print('Solved in 11 moves:', moves)
print(OUT / 'video.mp4')
