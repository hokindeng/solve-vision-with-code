from pathlib import Path
import subprocess
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent

def main():
    original = Image.open(ROOT / 'first_frame.png').convert('RGB')
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    # The numeric labels identify 35 (the first bar) as the minimum.
    # Put the highlight just outside the bar's existing black outline.
    corners = [(219, 770), (292, 770), (292, 875), (219, 875), (219, 770)]
    lengths = [abs(b[0]-a[0])+abs(b[1]-a[1]) for a,b in zip(corners, corners[1:])]
    perimeter = sum(lengths)
    process = subprocess.Popen([
        'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
        '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024',
        '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '0',
        '-preset', 'medium', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
        str(out / 'video.mp4')
    ], stdin=subprocess.PIPE)
    for index in range(48):
        frame = original.copy()
        if index:
            draw = ImageDraw.Draw(frame)
            remaining = perimeter * index / 47
            for a, b, length in zip(corners, corners[1:], lengths):
                if remaining <= 0:
                    break
                fraction = min(remaining / length, 1)
                end = (round(a[0] + (b[0]-a[0])*fraction),
                       round(a[1] + (b[1]-a[1])*fraction))
                draw.line([a, end], fill=(255, 0, 0), width=4)
                remaining -= length
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError('ffmpeg failed')

if __name__ == '__main__':
    main()
