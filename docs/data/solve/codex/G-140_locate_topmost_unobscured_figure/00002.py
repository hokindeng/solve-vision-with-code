from pathlib import Path
import subprocess
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path('/app')

def main():
    source = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    green = np.all(source == (25, 255, 25), axis=2)
    # Trace the unobscured triangle clockwise. Clip the stroke to its
    # interior so no background or other figure pixels are overwritten.
    vertices = np.array([(438, 412), (607, 704), (270, 704), (438, 412)], dtype=float)
    lengths = np.linalg.norm(np.diff(vertices, axis=0), axis=1)
    total = lengths.sum()
    (ROOT / 'output').mkdir(exist_ok=True)
    process = subprocess.Popen([
        'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
        '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024',
        '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '0',
        '-preset', 'slow', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
        str(ROOT / 'output/video.mp4')
    ], stdin=subprocess.PIPE)
    for frame_index in range(40):
        frame = source.copy()
        if frame_index:
            distance = total * min(frame_index / 38, 1)
            stroke = Image.new('L', (1024, 1024))
            draw = ImageDraw.Draw(stroke)
            for i, length in enumerate(lengths):
                amount = min(distance, length)
                if amount <= 0:
                    break
                start = vertices[i]
                end = start + (vertices[i + 1] - start) * amount / length
                draw.line([tuple(start), tuple(end)], fill=255, width=11)
                draw.ellipse((start[0]-5,start[1]-5,start[0]+5,start[1]+5),fill=255)
                distance -= amount
            outline = (np.array(stroke) > 0) & green
            frame[outline] = (255, 0, 0)
            assert np.array_equal(frame[~green], source[~green])
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError('ffmpeg failed')

if __name__ == '__main__':
    main()
