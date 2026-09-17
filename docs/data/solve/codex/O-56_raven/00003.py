from pathlib import Path
import numpy as np
from PIL import Image
import subprocess

ROOT = Path('/app')
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
base = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
# Copy the matching bottom-left corner's artwork, preserving its exact pixels.
target = base.copy()
target[700:1000, 702:1002] = base[700:1000, 20:320]
question = np.zeros((1024, 1024), dtype=bool)
question[700:1000, 702:1002] = np.any(base[700:1000, 702:1002] != 255, axis=2)
art = np.zeros_like(question)
art[700:1000, 702:1002] = np.any(target[700:1000, 702:1002] != 255, axis=2)
y, x = np.indices(question.shape)
# A clockwise tracing motion begins at the top vertex of both diamonds.
cx = np.where(x < 852, 766, 937)
phase = ((np.arctan2(y - 852, x - cx) + np.pi / 2) % (2 * np.pi)) / (2 * np.pi)
cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-vcodec', 'rawvideo',
       '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16', '-i', '-', '-an',
       '-c:v', 'libx264', '-crf', '0', '-preset', 'medium', '-pix_fmt', 'yuv420p',
       '-movflags', '+faststart', str(OUT / 'video.mp4')]
with subprocess.Popen(cmd, stdin=subprocess.PIPE) as proc:
    for i in range(35):
        frame = base.copy()
        fade = min(1.0, i / 11.0)
        frame[question] = np.rint(base[question] * (1 - fade) + 255 * fade).astype(np.uint8)
        progress = np.clip((i - 11) / 23.0, 0, 1)
        if progress > 0:
            visible = art & (phase <= progress)
            frame[visible] = target[visible]
        if i == 34:
            frame = target.copy()
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError('ffmpeg failed')

if __name__ == '__main__':
    print(OUT / 'video.mp4')
