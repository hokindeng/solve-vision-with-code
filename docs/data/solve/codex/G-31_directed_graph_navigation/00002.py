from pathlib import Path
import subprocess
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    # The only route from the start follows the two outgoing arrows
    # (708,751) -> (304,745) -> (204,271).
    agent_mask = (original[:, :, 0] == 0) & (original[:, :, 1] == 0) & (original[:, :, 2] > 0)
    yy, xx = np.where(agent_mask)
    agent_pixels = original[yy, xx].copy()
    background = original.copy()
    background[agent_mask] = (0, 128, 0)
    start = np.array([708., 751.])
    intermediate = np.array([304., 745.])
    end = np.array([204., 271.])
    output = ROOT / 'output'
    output.mkdir(exist_ok=True)
    command = ['ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
               '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16', '-i', '-',
               '-an', '-c:v', 'libx264', '-preset', 'slow', '-crf', '0',
               '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output / 'video.mp4')]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    for frame_index in range(30):
        if frame_index <= 14:
            position = start + (intermediate - start) * (frame_index / 14)
        else:
            position = intermediate + (end - intermediate) * ((frame_index - 14) / 15)
        dx, dy = np.rint(position - start).astype(int)
        frame = background.copy()
        frame[yy + dy, xx + dx] = agent_pixels
        if frame_index == 0:
            assert np.array_equal(frame, original)
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    error = process.stderr.read().decode()
    if process.wait():
        raise RuntimeError(error)

if __name__ == '__main__':
    main()
