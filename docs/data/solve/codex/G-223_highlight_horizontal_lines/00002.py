from pathlib import Path
import math
import numpy as np
from PIL import Image, ImageDraw
import imageio.v2 as imageio

ROOT = Path('/app')

def main():
    original = Image.open(ROOT / 'first_frame.png').convert('RGB')
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    scale = 4
    # Tight oval circles keep each horizontal line clearly isolated.
    rings = [(614.5, 577.5, 105.5, 23.0), (385.0, 704.5, 103.0, 23.0)]
    with imageio.get_writer(out / 'video.mp4', fps=16, codec='libx264', pixelformat='yuv420p', macro_block_size=None, ffmpeg_params=['-crf', '18']) as writer:
        for frame in range(48):
            overlay = Image.new('RGBA', (1024*scale, 1024*scale), (0,0,0,0))
            draw = ImageDraw.Draw(overlay)
            for i, (cx,cy,rx,ry) in enumerate(rings):
                start = 3 + 22*i
                progress = max(0.0, min(1.0, (frame-start)/20.0))
                if progress <= 0:
                    continue
                points = []
                for j in range(max(2, int(360*progress)+1)):
                    angle = -math.pi/2 + 2*math.pi*progress*j/(max(2,int(360*progress)+1)-1)
                    points.append(((cx+rx*math.cos(angle))*scale, (cy+ry*math.sin(angle))*scale))
                draw.line(points, fill=(0,0,0,255), width=3*scale, joint='curve')
            overlay = overlay.resize(original.size, Image.Resampling.LANCZOS)
            result = Image.alpha_composite(original.convert('RGBA'), overlay).convert('RGB')
            writer.append_data(np.asarray(result))

if __name__ == '__main__':
    main()
