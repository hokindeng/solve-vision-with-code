from pathlib import Path
import subprocess
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output'

def main():
    OUT.mkdir(exist_ok=True)
    source = Image.open(ROOT / 'first_frame.png').convert('RGB')
    circles = [((350, 307), 336, (245, 130, 48)),
               ((834, 716), 176, (255, 105, 180))]
    command = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
               '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024',
               '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '0',
               '-preset', 'medium', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
               str(OUT / 'video.mp4')]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    for i in range(40):
        t = i / 39
        progress = t * t * (3 - 2 * t)
        frame = Image.new('RGB', source.size, (255, 255, 255))
        draw = ImageDraw.Draw(frame)
        for (x0, y0), radius, color in circles:
            x = round(x0 + (512 - x0) * progress)
            y = round(y0 + (512 - y0) * progress)
            draw.ellipse((x-radius, y-radius, x+radius, y+radius),
                         outline=color, width=8)
        if i == 0:
            assert frame.tobytes() == source.tobytes()
            frame = source
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError('ffmpeg failed')

if __name__ == '__main__':
    main()
