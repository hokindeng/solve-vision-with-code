from pathlib import Path
import subprocess
from PIL import Image, ImageDraw

ROOT = Path('/app')

def main():
    output = ROOT / 'output'
    output.mkdir(exist_ok=True)
    base = Image.open(ROOT / 'first_frame.png').convert('RGB')
    # The middle-left room contains the sofas, rug and coffee table.
    corners = [(66, 336), (507, 336), (507, 626), (66, 626), (66, 336)]
    lengths = [abs(b[0]-a[0]) + abs(b[1]-a[1]) for a,b in zip(corners, corners[1:])]
    command = ['ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
               '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16', '-i', '-',
               '-an', '-c:v', 'libx264', '-crf', '0', '-preset', 'medium',
               '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output/'video.mp4')]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    for index in range(28):
        frame = base.copy()
        remaining = sum(lengths) * index / 27
        draw = ImageDraw.Draw(frame)
        for a, b, length in zip(corners, corners[1:], lengths):
            if remaining <= 0:
                break
            fraction = min(remaining / length, 1)
            end = (round(a[0] + (b[0]-a[0])*fraction), round(a[1] + (b[1]-a[1])*fraction))
            draw.line([a, end], fill=(0, 180, 45), width=6)
            remaining -= length
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    error = proc.stderr.read()
    if proc.wait():
        raise RuntimeError(error.decode())

if __name__ == '__main__':
    main()
