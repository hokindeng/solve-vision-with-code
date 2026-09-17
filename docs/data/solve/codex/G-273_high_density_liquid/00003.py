"""Animate identical blocks in liquids of different densities."""
from pathlib import Path
import math
import subprocess
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
FPS = 16
FRAMES = 80

def position(t, surface, floating):
    # Shared gravitational acceleration up to first contact with each liquid.
    start = 94.0
    side = 65
    acceleration = 370.0
    elapsed = max(0.0, t - 0.25)
    contact = math.sqrt(2 * (surface - side - start) / acceleration)
    if elapsed <= contact:
        return start + 0.5 * acceleration * elapsed ** 2
    wet_time = elapsed - contact
    if floating:
        # Damped vertical motion settles with 60 percent of the block submerged.
        equilibrium = surface - 26.0
        offset = -39.0
        damping = 2.5
        frequency = 5.0
        initial_velocity = acceleration * contact
        coefficient = (initial_velocity + damping * offset) / frequency
        return equilibrium + math.exp(-damping * wet_time) * (
            offset * math.cos(frequency * wet_time)
            + coefficient * math.sin(frequency * wet_time))
    # The less dense liquids slow the descent; the blocks settle on the bottom.
    initial = surface - side
    duration = 4.5 - (contact + 0.25)
    progress = min(1.0, wet_time / duration)
    # Smooth deceleration into contact with the floor.
    return initial + (791.0 - initial) * (1.0 - (1.0 - progress) ** 1.65)

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    xs = [236, 480, 724]
    sprites = [original[94:159, x:x+65].copy() for x in xs]
    background = original.copy()
    for x in xs:
        background[94:159, x:x+65] = 255
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    command = ['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt',
               'rgb24', '-s', '1024x1024', '-r', str(FPS), '-i', '-',
               '-an', '-c:v', 'libx264', '-crf', '15', '-preset', 'medium',
               '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out/'video.mp4')]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    for frame in range(FRAMES):
        canvas = background.copy()
        for i, (x, surface) in enumerate(zip(xs, [568, 479, 507])):
            y = int(round(position(frame / FPS, surface, i == 0)))
            canvas[y:y+65, x:x+65] = sprites[i]
        if frame == 0:
            assert np.array_equal(canvas, original)
        process.stdin.write(canvas.tobytes())
    process.stdin.close()
    if process.wait():
        raise RuntimeError('Video encoding failed')

if __name__ == '__main__':
    main()
