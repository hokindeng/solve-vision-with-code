from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import imageio.v2 as imageio

ROOT = Path(__file__).resolve().parent

def main():
    base = Image.open(ROOT / 'first_frame.png').convert('RGB')
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    # The alternating sizes require the large yellow pentagon at bottom right.
    # Only the choice mark is added; the puzzle itself remains untouched.
    with imageio.get_writer(out / 'video.mp4', fps=16, codec='libx264',
                            pixelformat='yuv420p', quality=10,
                            macro_block_size=1) as writer:
        for frame in range(60):
            im = base.copy()
            if frame >= 17:
                progress = min(1.0, (frame - 17) / 32.0)
                if progress > 0:
                    scale = 4
                    overlay = Image.new('RGBA', (1024*scale, 1024*scale))
                    draw = ImageDraw.Draw(overlay)
                    box = tuple(v*scale for v in (786, 786, 928, 928))
                    draw.arc(box, start=-90, end=-90 + 360*progress,
                             fill=(230, 30, 40, 255), width=5*scale)
                    overlay = overlay.resize(base.size, Image.Resampling.LANCZOS)
                    im = Image.alpha_composite(im.convert('RGBA'), overlay).convert('RGB')
            writer.append_data(np.asarray(im))

if __name__ == '__main__':
    main()
