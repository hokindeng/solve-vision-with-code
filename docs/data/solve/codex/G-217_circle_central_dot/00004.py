from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
import subprocess

ROOT = Path(__file__).resolve().parent

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    mask = np.uint8(np.min(original, axis=2) < 80)
    count, labels, stats, centers = cv2.connectedComponentsWithStats(mask)
    dots = sorted([(centers[i], stats[i]) for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] > 100], key=lambda item: item[0][0])
    center, stat = dots[len(dots)//2]
    cx, cy = center
    radius = max(stat[2], stat[3])/2 + 5
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    process = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '0', '-preset', 'slow', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out/'video.mp4')], stdin=subprocess.PIPE)
    scale = 4
    for index in range(22):
        frame = original.copy()
        if index:
            overlay = Image.new('L', (1024*scale, 1024*scale), 0)
            draw = ImageDraw.Draw(overlay)
            angles = np.linspace(-np.pi/2, -np.pi/2 + 2*np.pi*index/21, max(2, int(360*index/21)))
            points = [((cx+radius*np.cos(a))*scale, (cy+radius*np.sin(a))*scale) for a in angles]
            draw.line(points, fill=255, width=3*scale, joint='curve')
            for x,y in [points[0],points[-1]]:
                draw.ellipse((x-1.5*scale,y-1.5*scale,x+1.5*scale,y+1.5*scale),fill=255)
            alpha = np.array(overlay.resize((1024,1024), Image.Resampling.LANCZOS), dtype=np.float32)/255
            selected = alpha > 0
            frame[selected] = np.round(original[selected]*(1-alpha[selected,None]) + np.array([255,0,0])*alpha[selected,None]).astype(np.uint8)
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    if process.wait():
        raise RuntimeError('ffmpeg failed')
    print(f'Circled dot {len(dots)//2+1} of {len(dots)} at ({cx:.2f}, {cy:.2f})')

if __name__ == '__main__':
    main()
