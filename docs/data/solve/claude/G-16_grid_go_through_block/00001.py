import subprocess
import numpy as np
from PIL import Image, ImageDraw

W = H = 1024
FPS = 16
N_FRAMES = 102
ORIGIN, PITCH = 47, 93          # grid lines at 47,140,... (2px wide)
CELL_INNER = (49, 138)          # inner extent of first cell (relative offset 2..91)
AGENT_R = 23                    # ellipse bbox half-size (442..488 around 465)
FILL, OUTLINE = (255, 165, 0), (200, 120, 0)

# Cells (row, col) read from first_frame.png
START, END = (0, 4), (1, 6)
BLUE, PURPLE, PINK, YELLOW = (5, 3), (5, 2), (7, 5), (2, 8)
TARGETS = [BLUE, PURPLE, PINK, YELLOW, END]


def center(cell):
    r, c = cell
    return (ORIGIN + PITCH * c + 46, ORIGIN + PITCH * r + 46)


def shortest_path(a, b):
    """Manhattan path: move vertically first, then horizontally (4-neighbour moves)."""
    (r, c), (r2, c2) = a, b
    path = []
    while r != r2:
        r += 1 if r2 > r else -1
        path.append((r, c))
    while c != c2:
        c += 1 if c2 > c else -1
        path.append((r, c))
    return path


def build_route():
    route = [START]
    for t in TARGETS:
        route += shortest_path(route[-1], t)
    return route


def make_background(first):
    """Remove the agent from the start cell so it can be redrawn anywhere."""
    bg = first.copy()
    x0, y0 = ORIGIN + PITCH * START[1] + 2, ORIGIN + PITCH * START[0] + 2
    bg[y0:y0 + 90, x0:x0 + 90] = (50, 200, 50)
    return bg


def draw_agent(bg, pos):
    im = Image.fromarray(bg.copy())
    d = ImageDraw.Draw(im)
    x, y = pos
    d.ellipse([x - AGENT_R, y - AGENT_R, x + AGENT_R, y + AGENT_R],
              fill=FILL, outline=OUTLINE, width=2)
    return np.array(im)


def main():
    first = np.array(Image.open('/app/first_frame.png').convert('RGB'))
    bg = make_background(first)
    route = build_route()
    n_moves = len(route) - 1                      # 23 moves
    hold_start = 2
    frames_per_move = 4
    move_frames = n_moves * frames_per_move       # 92
    hold_end = N_FRAMES - hold_start - move_frames

    positions = []
    for _ in range(hold_start):
        positions.append(center(route[0]))
    for i in range(n_moves):
        (x0, y0), (x1, y1) = center(route[i]), center(route[i + 1])
        for k in range(1, frames_per_move + 1):
            t = k / frames_per_move
            positions.append((round(x0 + (x1 - x0) * t), round(y0 + (y1 - y0) * t)))
    for _ in range(hold_end):
        positions.append(center(route[-1]))
    assert len(positions) == N_FRAMES

    ff = subprocess.Popen([
        'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '12', '-preset', 'slow',
        '-r', str(FPS), '/app/output/video.mp4'], stdin=subprocess.PIPE)
    for i, pos in enumerate(positions):
        frame = first if i == 0 else draw_agent(bg, pos)
        ff.stdin.write(frame.tobytes())
    ff.stdin.close()
    ff.wait()
    # sanity: our drawing at the start reproduces the first frame exactly
    diff = np.abs(draw_agent(bg, center(START)).astype(int) - first.astype(int)).sum()
    print('route moves:', n_moves, '| redraw-vs-first_frame pixel diff:', diff)


if __name__ == '__main__':
    main()
