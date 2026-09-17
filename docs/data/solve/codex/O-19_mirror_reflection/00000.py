from pathlib import Path
import math
import numpy as np
from PIL import Image, ImageDraw
import imageio.v2 as imageio

ROOT = Path(__file__).resolve().parent

def main():
    source = Image.open(ROOT / 'first_frame.png').convert('RGB')
    original = np.asarray(source).copy()
    clean = source.copy()
    d = ImageDraw.Draw(clean)
    d.rectangle((641, 545, 724, 578), fill='white')
    # Remove the angle mark, restoring the incident ray beneath it.
    d.rectangle((592, 554, 597, 555), fill='white')
    d.rectangle((592, 554, 594, 555), fill=(0, 0, 255))
    cleaned = np.asarray(clean).copy()
    origin = np.array([596., 595.])
    direction = np.array([math.sin(math.radians(6)), -math.cos(math.radians(6))])
    total_length = origin[1] / -direction[1]
    color = (74, 74, 255) # 71% blue-ray intensity composited on white.
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    with imageio.get_writer(out / 'video.mp4', fps=16, codec='libx264',
                            pixelformat='yuv420p', quality=10,
                            macro_block_size=None, ffmpeg_params=['-crf', '0']) as writer:
        for i in range(35):
            if i == 0:
                writer.append_data(original)
                continue
            fade = min(i / 5., 1.)
            base = np.rint(original * (1-fade) + cleaned * fade).astype(np.uint8)
            frame = Image.fromarray(base)
            draw = ImageDraw.Draw(frame)
            progress = (i / 34.)
            length = total_length * progress
            tip = origin + direction * length
            draw.line([tuple(origin), tuple(tip)], fill=color, width=2)
            # Arrowhead follows the growing ray, pointing away from the mirror.
            if length > 35:
                arrowtip = origin + direction * max(20, length - 14)
                back = arrowtip - direction * 23
                normal = np.array([-direction[1], direction[0]])
                draw.line([tuple(back + normal * 10), tuple(arrowtip),
                           tuple(back - normal * 10)], fill=color, width=2)
            writer.append_data(np.asarray(frame))

if __name__ == '__main__':
    main()
