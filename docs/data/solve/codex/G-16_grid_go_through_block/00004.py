from pathlib import Path
import subprocess
import numpy as np
from PIL import Image

ROOT = Path('/app')

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    # Preserve the exact supplied agent artwork, including its darker outline.
    mask = ((original[:,:,0] == 255) & (original[:,:,1] == 165) & (original[:,:,2] == 0)) | ((original[:,:,0] == 200) & (original[:,:,1] == 120) & (original[:,:,2] == 0))
    yy, xx = np.where(mask)
    sprite = original[yy, xx].copy()
    background = original.copy()
    background[yy, xx] = (50, 200, 50)
    targets = [(9,2), (7,2), (6,1), (3,3), (3,7), (0,7), (1,8)]
    route = [targets[0]]
    for tx, ty in targets[1:]:
        x, y = route[-1]
        while x != tx:
            x += 1 if tx > x else -1
            route.append((x,y))
        while y != ty:
            y += 1 if ty > y else -1
            route.append((x,y))
    out = ROOT / 'output' / 'video.mp4'
    out.parent.mkdir(exist_ok=True)
    proc = subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pixel_format','rgb24','-video_size','1024x1024','-framerate','16','-i','-','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(out)], stdin=subprocess.PIPE)
    for i in range(82):
        progress = min(i / 79, 1) * (len(route)-1)
        j = min(int(progress), len(route)-2)
        fraction = progress-j
        x = route[j][0] + fraction*(route[j+1][0]-route[j][0])
        y = route[j][1] + fraction*(route[j+1][1]-route[j][1])
        dx, dy = round(93*(x-9)), round(93*(y-2))
        frame = background.copy()
        frame[yy+dy,xx+dx] = sprite
        if i == 0:
            assert np.array_equal(frame,original)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError('ffmpeg failed')

if __name__ == '__main__':
    main()
