from pathlib import Path
import subprocess
import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
base = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
# The visible sizes are medium, small, large, medium, small.
# Copy the large example exactly into the last position of the cycle.
color = np.array([236, 72, 153], dtype=np.uint8)
source = np.all(base == color, axis=2).astype(np.uint8)
source[:, :402] = 0
source[:, 474:] = 0
mask = np.zeros_like(source)
mask[:, 445:] = source[:, :-445]
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contour = contours[0][:, 0, :]
# Start at the apex and trace the perimeter before filling the interior.
start = np.argmin(contour[:, 1])
contour = np.roll(contour, -start, axis=0)
contour = np.concatenate([contour, contour[:1]])
ygrid = np.indices(mask.shape)[0]
cmd = ['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
       '-s', '1024x1024', '-r', '16', '-i', '-', '-an', '-c:v', 'libx264',
       '-preset', 'slow', '-qp', '0', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
       str(OUT / 'video.mp4')]
proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
for frame_number in range(60):
    frame = base.copy()
    drawing = np.zeros_like(mask)
    if frame_number >= 6:
        progress = min(1.0, (frame_number - 5) / 26)
        count = max(2, int(progress * (len(contour) - 1)) + 1)
        cv2.polylines(drawing, [contour[:count]], False, 1, 2, cv2.LINE_8)
        drawing &= mask
    if frame_number >= 32:
        progress = min(1.0, (frame_number - 31) / 23)
        drawing |= (mask.astype(bool) & (ygrid <= 473 + 69 * progress)).astype(np.uint8)
    frame[drawing.astype(bool)] = color
    # Only the new pentagon's pixels may change in the source frames.
    assert np.array_equal(frame[mask == 0], base[mask == 0])
    if frame_number == 0:
        assert np.array_equal(frame, base)
    proc.stdin.write(frame.tobytes())
proc.stdin.close()
if proc.wait() != 0:
    raise RuntimeError('ffmpeg encoding failed')
