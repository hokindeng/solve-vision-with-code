from pathlib import Path
import subprocess
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent

def main():
    base = Image.open(ROOT / 'first_frame.png').convert('RGB')
    output = ROOT / 'output'
    output.mkdir(exist_ok=True)
    # Enclose the maximum point (62) and its numeric label.
    corners = [(529, 176), (576, 176), (576, 249), (529, 249), (529, 176)]
    lengths = [abs(b[0]-a[0]) + abs(b[1]-a[1]) for a, b in zip(corners, corners[1:])]
    total = sum(lengths)
    proc = subprocess.Popen([
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16', '-i', '-',
        '-an', '-c:v', 'libx264', '-crf', '0', '-preset', 'medium',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
        str(output / 'video.mp4')
    ], stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    for index in range(48):
        frame = base.copy()
        if index:
            draw = ImageDraw.Draw(frame)
            remaining = total * index / 47
            for a, b, length in zip(corners, corners[1:], lengths):
                if remaining <= 0:
                    break
                fraction = min(remaining / length, 1)
                end = (round(a[0] + (b[0]-a[0])*fraction), round(a[1] + (b[1]-a[1])*fraction))
                draw.line([a, end], fill=(255, 0, 0), width=4)
                remaining -= length
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError('ffmpeg failed')

if __name__ == '__main__':
    main()
