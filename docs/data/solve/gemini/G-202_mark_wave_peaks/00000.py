import cv2
import numpy as np
import imageio
from scipy.signal import find_peaks
import os

def main():
    img = cv2.imread('/app/first_frame.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    y_coords, x_coords = np.where(thresh > 0)
    unique_x = np.unique(x_coords)
    
    # Calculate the mean to find the smooth line center
    mean_y = []
    for x in unique_x:
        mean_y.append(np.mean(y_coords[x_coords == x]))
    
    mean_y = np.array(mean_y)
    
    # Peaks in the wave are local maxima, which means local minima in y-coordinates
    peaks_idx, _ = find_peaks(-mean_y, prominence=10)
    peaks = [(int(unique_x[idx]), int(mean_y[idx])) for idx in peaks_idx]
    
    # Sort peaks left to right
    peaks.sort(key=lambda p: p[0])
    
    num_frames = 10
    os.makedirs('/app/output', exist_ok=True)
    
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, macro_block_size=1, codec='libx264', pixelformat='yuv420p')
    
    for i in range(num_frames):
        frame = img.copy()
        
        # Pace the action evenly over the frames
        show = int(round((i / (num_frames - 1)) * len(peaks)))
        show = min(show, len(peaks))
            
        for p in range(show):
            px, py = peaks[p]
            # Red hollow outline (circle)
            cv2.circle(frame, (px, py), 25, (0, 0, 255), 3)
            # Solid red dot at its center
            cv2.circle(frame, (px, py), 5, (0, 0, 255), -1)
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_rgb)
        
    writer.close()

if __name__ == '__main__':
    main()
