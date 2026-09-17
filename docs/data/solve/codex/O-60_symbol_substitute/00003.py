from pathlib import Path
import subprocess
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent

def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    # Restrict the edit to the interior of the fourth symbol's panel.
    region = original[478:547, 531:600].copy()
    old_mask = np.any(region != 255, axis=2)
    # Reuse the exact reference heart, translated to the fourth panel.
    heart = original[53:112, 914:979].copy()
    target = np.full_like(region, 255)
    target[10:69, 2:67] = heart
    new_mask = np.any(target != 255, axis=2)
    output = ROOT / 'output'
    output.mkdir(exist_ok=True)
    command = ['ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
               '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16',
               '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'slow',
               '-crf', '0', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
               str(output / 'video.mp4')]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    for index in range(52):
        frame = original.copy()
        edited = region.copy()
        old_opacity = 1.0 - smoothstep(index / 25.0)
        new_opacity = smoothstep((index - 26.0) / 25.0)
        edited[old_mask] = np.rint(255 + (region[old_mask].astype(float) - 255) * old_opacity).astype(np.uint8)
        if new_opacity > 0:
            edited[new_mask] = np.rint(255 + (target[new_mask].astype(float) - 255) * new_opacity).astype(np.uint8)
        frame[478:547, 531:600] = edited
        if index == 0:
            assert np.array_equal(frame, original)
        outside = frame.copy()
        outside[478:547, 531:600] = original[478:547, 531:600]
        assert np.array_equal(outside, original)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    errors = proc.stderr.read().decode()
    if proc.wait() != 0:
        raise RuntimeError(errors)

if __name__ == '__main__':
    main()
