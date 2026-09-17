import cv2
import numpy as np
import os
import subprocess
from scipy.signal import find_peaks

def main():
    img = cv2.imread('/app/first_frame.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    non_white = np.where(gray < 250)

    y_by_x = {}
    for y, x in zip(non_white[0], non_white[1]):
        if x not in y_by_x:
            y_by_x[x] = []
        y_by_x[x].append(y)

    points = []
    for x in sorted(y_by_x.keys()):
        avg_y = np.mean(y_by_x[x])
        points.append((x, avg_y))

    xs = np.array([p[0] for p in points])
    ys = np.array([p[1] for p in points])

    peaks, _ = find_peaks(-ys, prominence=50)
    
    # Coordinates of peaks
    peak_coords = [(int(xs[p]), int(np.round(ys[p]))) for p in peaks]
    peak_coords.sort(key=lambda c: c[0])
    
    frames = []
    
    # Frame 0: original
    frames.append(img.copy())
    
    for i, peak in enumerate(peak_coords):
        for step in range(1, 4):
            frame = img.copy()
            # Draw previously completed peaks
            for j in range(i):
                cv2.circle(frame, peak_coords[j], 30, (0, 0, 255), 3, cv2.LINE_AA)
                cv2.circle(frame, peak_coords[j], 6, (0, 0, 255), -1, cv2.LINE_AA)
                
            # Draw current peak animating
            r = step * 10
            cv2.circle(frame, peak, r, (0, 0, 255), 3, cv2.LINE_AA)
            cv2.circle(frame, peak, 6, (0, 0, 255), -1, cv2.LINE_AA)
            
            frames.append(frame)
            
    os.makedirs('/app/output', exist_ok=True)
    
    for idx, f in enumerate(frames):
        cv2.imwrite(f'/app/output/frame_{idx:03d}.png', f)
        
    cmd = [
        'ffmpeg', '-y', '-framerate', '16', '-i', '/app/output/frame_%03d.png',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '/app/output/video.mp4'
    ]
    subprocess.run(cmd, check=True)
    
    for idx in range(len(frames)):
        os.remove(f'/app/output/frame_{idx:03d}.png')

if __name__ == '__main__':
    main()
