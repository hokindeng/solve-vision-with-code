from pathlib import Path
import numpy as np
from PIL import Image
import subprocess

ROOT = Path(__file__).resolve().parent

def main():
    original = Image.open(ROOT / 'first_frame.png').convert('RGB')
    # Preserve the exact book sprites, including their two-pixel outlines.
    boxes = [(754, 371, 787, 513), (792, 356, 825, 513)]
    sprites = [original.crop(box) for box in boxes]
    base = original.copy()
    for x0, y0, x1, y1 in boxes:
        base.paste((255, 255, 255), (x0, y0, x1, 512))
        base.paste((92, 60, 40), (x0, 512, x1, 513))
    destinations = [190, 304]

    def position(index, progress):
        x, y = boxes[index][:2]
        dest = destinations[index]
        lifted_y = 246 - sprites[index].height
        # Lift clear of the tallest book, carry left, then lower into the gap.
        if progress <= .27:
            t = progress / .27
            return x, round(y + (lifted_y - y) * t)
        if progress <= .73:
            t = (progress - .27) / .46
            return round(x + (dest - x) * t), lifted_y
        t = (progress - .73) / .27
        return dest, round(lifted_y + (y - lifted_y) * t)

    output = ROOT / 'output'
    output.mkdir(exist_ok=True)
    command = ['ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
               '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16', '-i', '-',
               '-an', '-c:v', 'libx264', '-crf', '0', '-preset', 'medium',
               '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output / 'video.mp4')]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    for frame_number in range(24):
        if frame_number == 0:
            frame = original.copy()
        else:
            frame = base.copy()
            progress = [min(1., frame_number / 11), max(0., min(1., (frame_number - 11) / 11))]
            for index in range(2):
                frame.paste(sprites[index], position(index, progress[index]))
        process.stdin.write(np.asarray(frame).tobytes())
    process.stdin.close()
    errors = process.stderr.read()
    if process.wait() != 0:
        raise RuntimeError(errors.decode())

if __name__ == '__main__':
    main()
