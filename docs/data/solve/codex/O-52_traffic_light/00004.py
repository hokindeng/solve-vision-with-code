from pathlib import Path
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path('/app')
FPS = 16

def main():
    original = Image.open(ROOT / 'first_frame.png').convert('RGB')
    base = np.asarray(original).copy()
    # Yellow initial states refer to the first yellow in R -> Y -> G -> Y.
    lights = [
        (512, 220, 450, 275, 0),
        (512, 804, 450, 859, 4),
        (804, 512, 742, 567, 7),
        (220, 512, 158, 567, 5),
    ]
    colors = [(255, 0, 0), (255, 204, 0), (0, 255, 0), (255, 204, 0)]
    # Reuse the actual raster digits wherever available to preserve styling.
    digits = {
        4: original.crop((451, 276, 574, 399)),
        1: original.crop((743, 568, 866, 691)),
        3: original.crop((159, 568, 282, 691)),
    }
    two = Image.new('RGB', (123, 123), 'white')
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 100)
    d = ImageDraw.Draw(two)
    box = d.textbbox((0, 0), '2', font=font)
    d.text(((123-(box[2]-box[0]))/2-box[0], 24-box[1]), '2', font=font, fill='black')
    digits[2] = two
    masks = []
    yy, xx = np.indices(base.shape[:2])
    for cx, cy, bx, by, offset in lights:
        # Modify only original solid lamp fill, retaining outlines and plates.
        region = (abs(xx-cx)<70) & (abs(yy-cy)<70)
        fill = np.all(base == (255, 0, 0), axis=2) | np.all(base == (255, 204, 0), axis=2)
        masks.append(region & fill)
    (ROOT / 'output').mkdir(exist_ok=True)
    command = ['ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
               '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', str(FPS), '-i', '-',
               '-an', '-c:v', 'libx264', '-preset', 'medium', '-crf', '12',
               '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(ROOT/'output/video.mp4')]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    for frame in range(112):
        seconds = min(frame // FPS, 6)
        pixels = base.copy()
        for light, mask in zip(lights, masks):
            cx, cy, bx, by, offset = light
            phase_time = (offset + seconds) % 16
            pixels[mask] = colors[phase_time // 4]
            if seconds:
                pixels[by+1:by+124, bx+1:bx+124] = np.asarray(digits[4-phase_time % 4])
        if frame == 0:
            assert np.array_equal(pixels, base)
        process.stdin.write(pixels.tobytes())
    process.stdin.close()
    err = process.stderr.read()
    if process.wait():
        raise RuntimeError(err.decode())

if __name__ == '__main__':
    main()
