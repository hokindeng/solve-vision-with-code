from pathlib import Path
import subprocess
from PIL import Image, ImageDraw

ROOT = Path('/app')

def main():
    base = Image.open(ROOT / 'first_frame.png').convert('RGB')
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    # The stove and sink identify the upper-left room as the kitchen.
    # Keep the highlight just inside its enclosing walls.
    corners = [(69, 70), (296, 70), (296, 560), (69, 560), (69, 70)]
    lengths = [abs(b[0]-a[0]) + abs(b[1]-a[1]) for a, b in zip(corners, corners[1:])]
    total = sum(lengths)
    command = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
               '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024',
               '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '0',
               '-preset', 'medium', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
               str(out / 'video.mp4')]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE)
    for index in range(28):
        frame = base.copy()
        if index:
            draw = ImageDraw.Draw(frame)
            remaining = total * index / 27
            for a, b, length in zip(corners, corners[1:], lengths):
                if remaining <= 0:
                    break
                fraction = min(1, remaining / length)
                endpoint = (round(a[0] + (b[0]-a[0])*fraction),
                            round(a[1] + (b[1]-a[1])*fraction))
                draw.line([a, endpoint], fill=(0, 180, 45), width=6)
                remaining -= length
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError('ffmpeg encoding failed')

if __name__ == '__main__':
    main()
