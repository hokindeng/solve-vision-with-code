from pathlib import Path
from collections import deque
import subprocess
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path('/app')
SIZE = 1024
N = 15

def main():
    original = Image.open(ROOT / 'first_frame.png').convert('RGB')
    base = np.array(original)
    centers = [int((i + .5) * SIZE / N) for i in range(N)]
    walkable = {(r, c) for r in range(N) for c in range(N)
                if max(original.getpixel((centers[c], centers[r]))) > 80}
    start, end = (0, 3), (13, 12)
    walkable.update((start, end))
    queue = deque([start])
    parent = {start: None}
    while queue:
        r, c = queue.popleft()
        if (r, c) == end:
            break
        for p in ((r-1,c), (r+1,c), (r,c-1), (r,c+1)):
            if p in walkable and p not in parent:
                parent[p] = (r,c)
                queue.append(p)
    assert end in parent
    cells = []
    p = end
    while p is not None:
        cells.append(p)
        p = parent[p]
    cells.reverse()
    points = [(centers[c], centers[r]) for r,c in cells]
    print(f'Route: {len(cells)-1} adjacent-cell steps')
    # Overlay only the route in the white corridors. Preserve all original
    # walls and the start/end markers exactly in the raster frames.
    white = np.all(base == 255, axis=2)
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    command = ['ffmpeg','-y','-loglevel','error','-f','rawvideo',
               '-vcodec','rawvideo','-pix_fmt','rgb24','-s','1024x1024',
               '-r','16','-i','-','-an','-c:v','libx264','-preset','slow',
               '-crf','12','-pix_fmt','yuv420p','-movflags','+faststart',
               str(out / 'video.mp4')]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE)
    for frame in range(60):
        if frame == 0:
            rendered = base
        else:
            progress = (frame / 59) * (len(points)-1)
            segment = min(int(progress), len(points)-2)
            t = progress - segment
            a,b = points[segment],points[segment+1]
            head = (round(a[0]+(b[0]-a[0])*t), round(a[1]+(b[1]-a[1])*t))
            route = points[:segment+1] + [head]
            layer = original.copy()
            draw = ImageDraw.Draw(layer)
            draw.line(route, fill=(43,198,47), width=12, joint='curve')
            for x,y in route:
                draw.ellipse((x-6,y-6,x+6,y+6),fill=(43,198,47))
            x,y = head
            draw.ellipse((x-17,y-17,x+17,y+17),fill=(43,198,47))
            rendered = np.array(layer)
            rendered[~white] = base[~white]
        proc.stdin.write(rendered.tobytes())
    proc.stdin.close()
    assert proc.wait() == 0

if __name__ == '__main__':
    main()
