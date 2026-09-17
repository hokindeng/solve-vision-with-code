from pathlib import Path
from PIL import Image, ImageDraw
import subprocess

ROOT = Path(__file__).resolve().parent

def main():
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    base = Image.open(ROOT / 'first_frame.png').convert('RGB')
    # The bathroom occupies the bottom-left room, containing the tub,
    # toilet, and washbasin. Keep the highlight just inside its walls.
    corners = [(69, 711), (366, 711), (366, 955), (69, 955), (69, 711)]
    lengths = [abs(b[0]-a[0])+abs(b[1]-a[1]) for a,b in zip(corners,corners[1:])]
    total = sum(lengths)
    proc = subprocess.Popen([
        'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
        '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024',
        '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'slow',
        '-crf', '0', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
        str(out / 'video.mp4')
    ], stdin=subprocess.PIPE)
    for frame in range(28):
        img = base.copy()
        if frame:
            remaining = total * frame / 27
            draw = ImageDraw.Draw(img)
            for a, b, length in zip(corners, corners[1:], lengths):
                if remaining <= 0:
                    break
                fraction = min(remaining / length, 1)
                end = (round(a[0] + (b[0]-a[0])*fraction),
                       round(a[1] + (b[1]-a[1])*fraction))
                draw.line([a, end], fill=(0, 180, 45), width=6)
                remaining -= length
        proc.stdin.write(img.tobytes())
    proc.stdin.close()
    if proc.wait():
        raise RuntimeError('ffmpeg failed')

if __name__ == '__main__':
    main()
