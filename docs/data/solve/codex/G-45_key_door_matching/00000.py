from pathlib import Path
from collections import deque
import subprocess
import numpy as np
from PIL import Image

ROOT = Path('/app')

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    green = (original[:,:,0] == 0) & (original[:,:,1] == 255) & (original[:,:,2] == 0)
    red_key = (original[:,:,0] == 255) & (original[:,:,1] == 0) & (original[:,:,2] == 0)
    red_key[:887] = False
    ys, xs = np.where(green)
    cx, cy = round(xs.mean()), round(ys.mean())
    base = original.copy()
    base[green] = 255
    centers = [int((i + .5) * 1024 / 15) for i in range(15)]
    grid = np.array([[original[y,x].max() > 0 for x in centers] for y in centers])
    def route(start, end):
        queue = deque([start])
        previous = {start: None}
        while queue:
            p = queue.popleft()
            if p == end:
                break
            for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                n = p[0]+dx, p[1]+dy
                if 0 <= n[0] < 15 and 0 <= n[1] < 15 and grid[n[1],n[0]] and n not in previous:
                    previous[n] = p
                    queue.append(n)
        points = []
        p = end
        while p is not None:
            points.append((centers[p[0]], centers[p[1]]))
            p = previous[p]
        return np.array(points[::-1], dtype=float)
    first = route((1,1), (7,13))
    second = route((7,13), (3,11))
    def position(points, fraction):
        lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        cumulative = np.r_[0, np.cumsum(lengths)]
        distance = np.clip(fraction,0,1) * cumulative[-1]
        segment = min(np.searchsorted(cumulative, distance, side='right') - 1, len(lengths)-1)
        return points[segment] + (points[segment+1]-points[segment]) * ((distance-cumulative[segment])/lengths[segment])
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    proc = subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pixel_format','rgb24','-video_size','1024x1024','-framerate','16','-i','-','-an','-c:v','libx264','-preset','medium','-crf','0','-pix_fmt','yuv420p','-movflags','+faststart',str(out/'video.mp4')], stdin=subprocess.PIPE)
    for frame in range(198):
        collected = frame >= 103
        if frame < 4:
            canvas = original.copy()
        else:
            if frame <= 103:
                pos = position(first, (frame-4)/99)
            elif frame <= 108:
                pos = first[-1]
            else:
                pos = position(second, (frame-108)/81)
            canvas = base.copy()
            if collected:
                canvas[red_key] = 255
            dx, dy = np.rint(pos - [cx, cy]).astype(int)
            # Translate the original circle mask without modifying any other pixels.
            canvas[ys+dy, xs+dx] = [0,255,0]
            assert np.all(np.any(base[ys+dy, xs+dx] != 0, axis=1)), 'Agent intersects wall'
        proc.stdin.write(canvas.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError('ffmpeg failed')

if __name__ == '__main__':
    main()
