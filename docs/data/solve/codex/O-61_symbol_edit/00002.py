from pathlib import Path
import subprocess
import numpy as np
from PIL import Image

ROOT = Path('/app')
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
base = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
xs = [44 + 105*i for i in range(9)]
# Preserve the original raster artwork by copying only cell interiors.
patches = [base[465:560, x+1:x+96].copy() for x in xs[:5]]
diamond = patches[4]

def render(sequence):
    frame = base.copy()
    for i, x in enumerate(xs):
        frame[465:560, x+1:x+96] = sequence[i] if i < len(sequence) else 255
    return frame

sequence = patches.copy()
states = [base]
for position in (3, 6, 8):
    sequence.insert(position-1, diamond.copy())
    states.append(render(sequence))

# Each insertion has a short, smooth reveal, with time to inspect each state.
transitions = [(10, 23), (33, 46), (57, 70)]
command = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
           '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16', '-i', '-',
           '-an', '-c:v', 'libx264', '-preset', 'slow', '-crf', '0',
           '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
           str(OUT / 'video.mp4')]
process = subprocess.Popen(command, stdin=subprocess.PIPE)
for frame_number in range(76):
    frame = states[0]
    for j, (start, end) in enumerate(transitions):
        if frame_number >= end:
            frame = states[j+1]
        elif frame_number >= start:
            alpha = (frame_number-start)/(end-start)
            alpha = alpha*alpha*(3-2*alpha)
            frame = np.rint(states[j].astype(float)*(1-alpha) +
                            states[j+1].astype(float)*alpha).astype(np.uint8)
            break
        else:
            break
    process.stdin.write(frame.tobytes())
process.stdin.close()
if process.wait() != 0:
    raise RuntimeError('ffmpeg failed')
