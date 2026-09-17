from pathlib import Path
import subprocess
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent

def main():
    original = Image.open(ROOT / 'first_frame.png').convert('RGB')
    pixels = np.array(original)
    # The central flat-color region is the innermost concentric square.
    center_color = pixels[original.height // 2, original.width // 2]
    ys, xs = np.where(np.all(pixels == center_color, axis=2))
    left, right, top, bottom = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    width = 8
    # Keep the complete outline inside the target square's boundary.
    inset = width // 2
    points = [(left+inset, top+inset), (right-inset, top+inset),
              (right-inset, bottom-inset), (left+inset, bottom-inset),
              (left+inset, top+inset)]
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    command = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
               '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024',
               '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-preset',
               'slow', '-crf', '0', '-pix_fmt', 'yuv420p', '-movflags',
               '+faststart', str(out / 'video.mp4')]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    for frame_number in range(85):
        frame = original.copy()
        draw = ImageDraw.Draw(frame)
        # Four successive steps: top, right, bottom, then left edge.
        progress = min(4.0, max(0.0, (frame_number - 4) / 18.0))
        for edge in range(4):
            fraction = min(1.0, max(0.0, progress - edge))
            if fraction > 0:
                start, end = points[edge], points[edge+1]
                tip = tuple(round(a + (b-a)*fraction) for a,b in zip(start,end))
                draw.line([start, tip], fill=(0, 0, 255), width=width)
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError('ffmpeg failed')

if __name__ == '__main__':
    main()
