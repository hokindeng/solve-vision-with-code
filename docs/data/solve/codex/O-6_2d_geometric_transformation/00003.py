from pathlib import Path
import subprocess
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path('/app')
OUTPUT = ROOT / 'output/video.mp4'

def main():
    source = Image.open(ROOT / 'first_frame.png').convert('RGB')
    original = np.array(source)
    fill = (192, 152, 55)
    edge = (50, 50, 50)
    moving = np.all(original == fill, axis=2) | np.all(original == edge, axis=2)
    background = original.copy()
    background[moving] = (240, 240, 240)
    marker = np.all(original == (0, 0, 0), axis=2) | np.all(original == (255, 255, 255), axis=2)
    center = np.array([239., 472.])
    vertices = np.array([[239., 472.], [283., 467.], [337., 552.], [248., 561.]])
    angle = np.deg2rad(143.11491377425003)
    OUTPUT.parent.mkdir(exist_ok=True)
    proc = subprocess.Popen([
        'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16', '-i', '-',
        '-an', '-c:v', 'libx264', '-preset', 'slow', '-crf', '17',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(OUTPUT)
    ], stdin=subprocess.PIPE)
    for frame in range(70):
        if frame <= 4:
            result = original
        else:
            progress = np.clip((frame - 4) / 60., 0., 1.)
            progress = progress * progress * (3. - 2. * progress)
            a = angle * progress
            rotation = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
            points = (vertices - center) @ rotation.T + center
            canvas = Image.fromarray(background.copy())
            draw = ImageDraw.Draw(canvas)
            draw.polygon([tuple(p) for p in points], fill=fill, outline=edge, width=1)
            result = np.array(canvas)
            result[marker] = original[marker]
        proc.stdin.write(result.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError('Video encoding failed')

if __name__ == '__main__':
    main()
