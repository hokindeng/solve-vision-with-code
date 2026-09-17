from pathlib import Path
import cv2
import numpy as np
from scipy.interpolate import PchipInterpolator
import subprocess

ROOT = Path('/app')

def main():
    original = cv2.imread(str(ROOT / 'first_frame.png'))
    rgb = original[:, :, ::-1]
    ball = ((rgb[:, :, 0] > rgb[:, :, 1]) & (rgb[:, :, 1] > rgb[:, :, 2])).astype(np.float32)
    background = original.copy()
    background[ball > 0] = 255
    # Keep every visible platform pixel exactly as supplied.
    blue = ((original[:, :, 0] > 100) & (original[:, :, 2] < 100) & (original[:, :, 1] > 50)).astype(np.uint8)
    n, labels, stats, centers = cv2.connectedComponentsWithStats(blue)
    platforms = sorted([centers[i] for i in range(1, n) if stats[i, 4] > 10], key=lambda p: p[0])
    # Centers follow the upper faces, with the ball radius accounted for.
    points = np.array([[350., 550.]] + [[p[0], p[1] - 46.] for p in platforms])
    knots = np.arange(len(points), dtype=float)
    curve_x = PchipInterpolator(knots, points[:, 0])
    curve_y = PchipInterpolator(knots, points[:, 1])
    foreground = original.astype(np.float32) * ball[:, :, None]
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    proc = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-vcodec', 'rawvideo', '-pix_fmt', 'bgr24', '-s', '1024x1024', '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '15', '-preset', 'slow', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out / 'video.mp4')], stdin=subprocess.PIPE)
    for frame in range(64):
        if frame == 0:
            result = original
        else:
            t = min(frame / 61., 1.)
            progress = (3*t*t - 2*t*t*t) * (len(points)-1)
            # Small arcs connect successive landings without interrupting forward motion.
            lift = 1.5 * np.sin(np.pi * (progress % 1.))**2
            x, y = float(curve_x(progress)), float(curve_y(progress)) - lift
            transform = np.float32([[1, 0, x-350.], [0, 1, y-550.]])
            alpha = cv2.warpAffine(ball, transform, (1024, 1024), flags=cv2.INTER_LINEAR)
            color = cv2.warpAffine(foreground, transform, (1024, 1024), flags=cv2.INTER_LINEAR)
            result = np.clip(color + background.astype(np.float32)*(1-alpha[:, :, None]), 0, 255).astype(np.uint8)
        proc.stdin.write(result.tobytes())
    proc.stdin.close()
    if proc.wait():
        raise RuntimeError('Video encoding failed')

if __name__ == '__main__':
    main()
