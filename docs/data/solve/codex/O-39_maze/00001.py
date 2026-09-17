from pathlib import Path
from collections import deque
import numpy as np
from PIL import Image, ImageDraw
import subprocess

ROOT = Path('/app')

def main():
    original = Image.open(ROOT / 'first_frame.png').convert('RGB')
    arr = np.array(original)
    n = 15
    def center(cell):
        r,c = cell
        return (int((c+.5)*1024/n), int((r+.5)*1024/n))
    walkable = set()
    for r in range(n):
        for c in range(n):
            x,y = center((r,c))
            if np.median(arr[y-5:y+6,x-5:x+6].max(axis=2)) > 80:
                walkable.add((r,c))
    start, end = (1,4), (12,10)
    walkable.update((start,end))
    prev = {start: None}
    queue = deque([start])
    while queue:
        p = queue.popleft()
        if p == end:
            break
        for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)):
            q = (p[0]+dr,p[1]+dc)
            if q in walkable and q not in prev:
                prev[q] = p
                queue.append(q)
    assert end in prev
    path = []
    p = end
    while p is not None:
        path.append(p)
        p = prev[p]
    path.reverse()
    points = [center(p) for p in path]
    # Only white pixels may receive the route. Existing markers remain intact.
    white = np.all(arr == 255, axis=2)
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    proc = subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(out/'video.mp4')], stdin=subprocess.PIPE)
    for frame in range(63):
        result = arr.copy()
        if frame:
            progress = (len(points)-1)*frame/62
            k = min(int(progress),len(points)-1)
            drawn = points[:k+1]
            if k < len(points)-1:
                f = progress-k
                a,b = points[k],points[k+1]
                drawn.append((round(a[0]+f*(b[0]-a[0])), round(a[1]+f*(b[1]-a[1]))))
            layer = Image.new('L',original.size)
            d = ImageDraw.Draw(layer)
            d.line(drawn,fill=255,width=14,joint='curve')
            for x,y in drawn:
                d.ellipse((x-7,y-7,x+7,y+7),fill=255)
            x,y = drawn[-1]
            d.ellipse((x-15,y-15,x+15,y+15),fill=255)
            mask = (np.array(layer)>0)&white
            result[mask] = (45,195,55)
        proc.stdin.write(result.tobytes())
    proc.stdin.close()
    assert proc.wait() == 0
    print(f'Created video with 63 frames; route contains {len(path)} cells.')

if __name__ == '__main__':
    main()
