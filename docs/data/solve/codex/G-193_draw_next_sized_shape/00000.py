from pathlib import Path
import numpy as np
from PIL import Image
import imageio.v2 as imageio

ROOT = Path('/app')

def main():
    base = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    # The sequence is large, medium, small, large, medium: repeat small.
    # Copy the existing small circle exactly, centered within the answer box.
    patch = base[493:532, 419:458].copy()
    mask = np.any(patch != 255, axis=2)
    yy, xx = np.mgrid[-19:20, -19:20]
    radius = np.sqrt(xx*xx + yy*yy)
    angle = (np.arctan2(yy, xx) + np.pi/2) % (2*np.pi)
    output = ROOT / 'output'
    output.mkdir(exist_ok=True)
    with imageio.get_writer(output / 'video.mp4', fps=16, codec='libx264',
                            pixelformat='yuv420p', macro_block_size=1,
                            ffmpeg_params=['-crf', '0']) as writer:
        for frame in range(60):
            canvas = base.copy()
            # Pause to observe the repeating sizes, trace the circumference,
            # then fill the circle from the boundary inward.
            if frame >= 10:
                if frame < 32:
                    progress = (frame - 10) / 21
                    visible = mask & (radius >= 16.5) & (angle <= progress*2*np.pi)
                else:
                    progress = min(1., (frame-32)/22)
                    visible = mask & (radius >= 16.5*(1-progress))
                region = canvas[493:532, 867:906]
                region[visible] = patch[visible]
            writer.append_data(canvas)

if __name__ == '__main__':
    main()
