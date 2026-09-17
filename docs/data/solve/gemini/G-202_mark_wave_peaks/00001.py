import cv2
import numpy as np
import imageio
from scipy.signal import find_peaks
import os

def generate_video():
    first_frame_path = '/app/first_frame.png'
    output_dir = '/app/output'
    output_path = os.path.join(output_dir, 'video.mp4')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    img = cv2.imread(first_frame_path)
    if img is None:
        raise ValueError(f"Could not read {first_frame_path}")
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    wave_pixels = []
    for x in range(w):
        col = gray[:, x]
        ys = np.where(col < 128)[0]
        if len(ys) > 0:
            wave_pixels.append((x, np.mean(ys)))

    xs = np.array([p[0] for p in wave_pixels])
    ys = np.array([p[1] for p in wave_pixels])

    peaks, _ = find_peaks(-ys, prominence=5)
    
    frames = []
    frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    current_img = img.copy()
    
    radii = [10, 20, 30]
    
    for i, p in enumerate(peaks):
        cx, cy = int(xs[p]), int(ys[p])
        
        for r in radii:
            frame = current_img.copy()
            cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
            cv2.circle(frame, (cx, cy), r, (0, 0, 255), 3)
            
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            if r == radii[-1]:
                current_img = frame
                
    writer = imageio.get_writer(
        output_path, 
        fps=16, 
        codec='libx264', 
        format='FFMPEG', 
        pixelformat='yuv420p'
    )
    
    for f in frames:
        writer.append_data(f)
        
    writer.close()

if __name__ == '__main__':
    generate_video()
