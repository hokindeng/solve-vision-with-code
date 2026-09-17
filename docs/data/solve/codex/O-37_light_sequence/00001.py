from pathlib import Path
import subprocess
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    final = original.copy()
    # Each cell contains one original light. Reuse its exact disk silhouette.
    centers = [153.6, 256, 358.4, 460.8, 563.2, 665.6, 768, 870.4]
    yy, xx = np.indices(original.shape[:2])
    gray = np.all(original == (128,128,128), axis=2)
    dark = np.all(original == (64,64,64), axis=2)
    gold = np.all(original == (255,215,0), axis=2)
    orange = np.all(original == (255,165,0), axis=2)
    halo = np.all(original == (255,243,179), axis=2)
    regions = []
    for i, cx in enumerate(centers):
        cell = (xx >= cx-40) & (xx <= cx+40)
        if i % 2 == 0:
            # The outer glow is the same pale gold as in the source image.
            glow = ((xx-cx)**2 + (yy-512)**2 <= 32.5**2) & cell
            final[glow] = (255,243,179)
            final[gray & cell] = (255,215,0)
            final[dark & cell] = (255,165,0)
        else:
            final[halo & cell] = (255,255,255)
            final[gold & cell] = (128,128,128)
            final[orange & cell] = (64,64,64)
        region = cell & np.any(final != original, axis=2)
        if region.any():
            regions.append(region)
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    cmd = ['ffmpeg','-y','-v','error','-f','rawvideo','-pixel_format','rgb24',
           '-video_size','1024x1024','-framerate','16','-i','-',
           '-an','-c:v','libx264','-preset','slow','-crf','0',
           '-pix_fmt','yuv420p','-movflags','+faststart',str(out/'video.mp4')]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for frame_number in range(35):
        frame = original.copy()
        for j, region in enumerate(regions):
            start = 1 + j * 4
            alpha = np.clip((frame_number-start)/3, 0, 1)
            frame[region] = np.rint(original[region].astype(float)*(1-alpha) + final[region]*alpha).astype(np.uint8)
        if frame_number == 0:
            assert np.array_equal(frame, original)
        unchanged = ~np.any(final != original, axis=2)
        assert np.array_equal(frame[unchanged], original[unchanged])
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError('ffmpeg failed')

if __name__ == '__main__':
    main()
