from pathlib import Path
import subprocess
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path('/app')

def main():
    original = Image.open(ROOT / 'first_frame.png').convert('RGB')
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    # The only square object inside the left square region.
    points = [(279, 474), (347, 474), (347, 542), (279, 542), (279, 474)]
    command = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
               '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024',
               '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '0',
               '-preset', 'medium', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
               str(out / 'video.mp4')]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    for frame_number in range(40):
        frame = original.copy()
        if frame_number:
            draw = ImageDraw.Draw(frame)
            remaining = 272 * frame_number / 39
            for start, end in zip(points, points[1:]):
                length = min(68, remaining)
                if length <= 0:
                    break
                endpoint = (round(start[0] + (end[0]-start[0])*length/68),
                            round(start[1] + (end[1]-start[1])*length/68))
                draw.line([start, endpoint], fill=(0, 180, 0), width=4)
                remaining -= 68
        process.stdin.write(np.asarray(frame).tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError('Video encoding failed')

if __name__ == '__main__':
    main()
