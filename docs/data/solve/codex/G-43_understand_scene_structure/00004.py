from pathlib import Path
import subprocess
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent

def main():
    base = Image.open(ROOT / 'first_frame.png').convert('RGB')
    out = ROOT / 'output' / 'video.mp4'
    out.parent.mkdir(parents=True, exist_ok=True)
    # The upper-left room contains the toilet, sink and bathtub.
    # Trace just inside its surrounding walls, preserving the fixtures.
    corners = [(68, 68), (401, 68), (401, 387), (68, 387), (68, 68)]
    lengths = [abs(b[0]-a[0]) + abs(b[1]-a[1]) for a,b in zip(corners, corners[1:])]
    encoder = subprocess.Popen([
        'ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', '1024x1024', '-r', '16', '-i', '-', '-an',
        '-c:v', 'libx264', '-preset', 'slow', '-crf', '0',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out)
    ], stdin=subprocess.PIPE)
    for index in range(28):
        frame = base.copy()
        if index:
            draw = ImageDraw.Draw(frame)
            remaining = sum(lengths) * index / 27
            for start, end, length in zip(corners, corners[1:], lengths):
                if remaining <= 0:
                    break
                fraction = min(remaining / length, 1)
                tip = (round(start[0] + fraction*(end[0]-start[0])),
                       round(start[1] + fraction*(end[1]-start[1])))
                draw.line([start, tip], fill=(0, 180, 45), width=6)
                remaining -= length
        encoder.stdin.write(frame.tobytes())
    encoder.stdin.close()
    if encoder.wait():
        raise RuntimeError('Video encoding failed')

if __name__ == '__main__':
    main()
