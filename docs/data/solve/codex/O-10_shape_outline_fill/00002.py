from pathlib import Path
import subprocess
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    # Match the reference's three-pixel marks, one-pixel gaps, and four-pixel stroke.
    marks = []
    for x in range(785, 944, 4):
        marks.append((slice(687, 691), slice(x, min(x + 3, 944))))
    for y in range(689, 848, 4):
        marks.append((slice(y, min(y + 3, 848)), slice(942, 946)))
    for x in reversed(range(785, 944, 4)):
        marks.append((slice(847, 851), slice(x, min(x + 3, 944))))
    for y in reversed(range(689, 848, 4)):
        marks.append((slice(y, min(y + 3, 848)), slice(782, 786)))
    purple = np.array([152, 114, 229], dtype=float)
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
           '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024',
           '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '0',
           '-preset', 'medium', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
           str(out / 'video.mp4')]
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for frame_index in range(60):
        frame = original.copy()
        # Erase only the answer placeholder as the replacement is revealed.
        fade = min(1.0, frame_index / 16.0)
        region = original[740:798, 840:889].astype(float)
        frame[740:798, 840:889] = np.rint(region * (1 - fade) + 255 * fade).astype(np.uint8)
        progress = np.clip((frame_index - 10) / 46, 0, 1) * len(marks)
        for i, (ys, xs) in enumerate(marks):
            alpha = np.clip(progress - i, 0, 1)
            if alpha:
                frame[ys, xs] = np.rint(255 * (1 - alpha) + purple * alpha).astype(np.uint8)
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError('ffmpeg failed')

if __name__ == '__main__':
    main()
