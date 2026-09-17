"""Generate the analogy-completion video: thick solid outline -> thin dotted outline.
Row 1 shows a triangle whose 8px solid outline becomes a 4px dotted outline.
Row 2 applies the same style change to the square: the '?' fades out and a thin
dotted square outline is traced in at the mirrored position."""
import subprocess, os
import numpy as np
from PIL import Image

SRC = '/app/first_frame.png'
OUT = '/app/output/video.mp4'
N_FRAMES = 60
FPS = 16
PURPLE = np.array([152, 114, 229], dtype=np.uint8)

base = np.array(Image.open(SRC).convert('RGB'))
H, W, _ = base.shape

# --- geometry -------------------------------------------------------------
# Left square outer bbox: x 76..244, y 684..852 (8px stroke, path centre ~79.5 / 240.5).
# Row 1 shift left->right is +704 px; dotted stroke is 4px wide, 3 on / 1 off.
DX = 704
SHIFT_X0, SHIFT_X1 = 76 + DX, 244 + DX   # outer bounds of the stroke, inclusive
Y0, Y1 = 684, 852
STROKE = 4
DASH_ON, DASH_PERIOD = 3, 4

def path_points():
    """Pixel-step walk around the square path centre, clockwise from top-left."""
    # centre lines of the 4px stroke: left col band 782..785 -> cx 784 (draw cx-2..cx+1)
    # source strokes: cols 76..83 / 237..244, rows 685..692 / 844..851 (centres 79.5, 240.5, 688.5, 847.5)
    cx0, cx1 = 784, 945                        # draw cx-2..cx+1 -> 782..785, 943..946
    cy0, cy1 = 689, 848                        # draw cy-2..cy+1 -> 687..690, 846..849
    pts = []
    for x in range(cx0, cx1): pts.append((x, cy0, 'h'))
    for y in range(cy0, cy1): pts.append((cx1, y, 'v'))
    for x in range(cx1, cx0, -1): pts.append((x, cy1, 'h'))
    for y in range(cy1, cy0, -1): pts.append((cx0, y, 'v'))
    return pts

PTS = path_points()
PERIM = len(PTS)

def outline_mask(progress):
    """Mask of the dotted outline traced up to `progress` (0..1) of the perimeter."""
    m = np.zeros((H, W), dtype=bool)
    n = int(round(progress * PERIM))
    for d in range(n):
        if d % DASH_PERIOD >= DASH_ON:
            continue
        x, y, o = PTS[d]
        if o == 'h':
            m[y - 2:y + 2, x] = True
        else:
            m[y, x - 2:x + 2] = True
    return m

# '?' glyph mask: non-white, non-purple pixels inside the bottom-right cell
# restricted to the glyph's own box so the row-2 arrowhead (which crosses x=512) is untouched
QX0, QX1, QY0, QY1 = 820, 910, 720, 815
qmask = np.zeros((H, W), dtype=bool)
sub = base[QY0:QY1, QX0:QX1].astype(int)
notwhite = (sub.sum(2) < 255 * 3 - 3)
notpurple = np.abs(sub - PURPLE.astype(int)).sum(2) > 40
qmask[QY0:QY1, QX0:QX1] = notwhite & notpurple

def smooth(t):
    return t * t * (3 - 2 * t)

def frame(i):
    t = i / (N_FRAMES - 1)
    img = base.copy()
    # Phase 1: fade the '?' to white over the first ~third
    fade = smooth(min(1.0, t / 0.35))
    if fade > 0:
        q = img[qmask].astype(float)
        img[qmask] = np.clip(q + (255 - q) * fade, 0, 255).astype(np.uint8)
    # Phase 2: trace dotted square outline from ~20% to ~92% of the clip
    prog = smooth(min(1.0, max(0.0, (t - 0.20) / 0.72)))
    if prog > 0:
        img[outline_mask(prog)] = PURPLE
    return img

def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    frames = [frame(i) for i in range(N_FRAMES)]
    assert np.array_equal(frames[0], base)
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
           '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
           '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '12', '-preset', 'slow', OUT]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(f.tobytes())
    p.stdin.close(); p.wait()
    assert p.returncode == 0
    Image.fromarray(frames[-1]).save('/app/output/last_frame.png')
    print('wrote', OUT, 'frames', len(frames), 'perimeter', PERIM)

if __name__ == '__main__':
    main()
