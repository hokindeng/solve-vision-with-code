from pathlib import Path
import subprocess
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent

def main():
    base = Image.open(ROOT / 'first_frame.png').convert('RGB')
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    process = subprocess.Popen([
        'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
        '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024',
        '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '0',
        '-preset', 'medium', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
        str(out / 'video.mp4')
    ], stdin=subprocess.PIPE)
    # The sole circle inside the large circular enclosure is the pink disk.
    # Supersample a local overlay so pixels elsewhere remain identical.
    scale = 4
    box = (240, 363, 310, 433)
    for frame in range(40):
        result = base.copy()
        if frame:
            mask = Image.new('L', (70 * scale, 70 * scale), 0)
            draw = ImageDraw.Draw(mask)
            bounds = tuple(round(v * scale) for v in (5.5, 5.5, 63, 63))
            draw.arc(bounds, -90, -90 + 360 * frame / 39,
                     fill=255, width=3 * scale)
            mask = mask.resize((70, 70), Image.Resampling.LANCZOS)
            patch = base.crop(box)
            patch.paste((0, 170, 45), (0, 0, 70, 70), mask)
            result.paste(patch, box)
        process.stdin.write(np.asarray(result).tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError('Video encoding failed')

if __name__ == '__main__':
    main()
