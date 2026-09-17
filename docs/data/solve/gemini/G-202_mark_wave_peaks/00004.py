import cv2
import numpy as np
from scipy.signal import find_peaks
import os

def main():
    img = cv2.imread('first_frame.png')
    if img is None:
        raise ValueError("Could not read first_frame.png")
    
    # Find black pixels
    y, x = np.where((img == [0, 0, 0]).all(axis=2))
    
    # Calculate mean y for each x
    unique_x = np.unique(x)
    mean_y = []
    for ux in unique_x:
        mean_y.append(np.mean(y[x == ux]))
    mean_y = np.array(mean_y)
    
    # Find local minima in y (peaks of the wave)
    peaks_idx, _ = find_peaks(-mean_y, prominence=10)
    
    peaks = []
    for idx in peaks_idx:
        px = int(unique_x[idx])
        py = int(mean_y[idx])
        peaks.append((px, py))
        
    # Sort peaks from left to right (should already be sorted by x)
    peaks.sort(key=lambda p: p[0])
    
    # Animation settings
    num_frames = 10
    out_dir = 'output'
    os.makedirs(out_dir, exist_ok=True)
    
    # Prepare video writer
    h, w, _ = img.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = os.path.join(out_dir, 'video.mp4')
    
    # We will use imageio for writing since cv2 VideoWriter sometimes has codec issues in minimal environments
    import imageio
    writer = imageio.get_writer(out_path, fps=16, codec='libx264', pixelformat='yuv420p')
    
    dot_radius = 6
    outline_radius = 24
    color = (255, 0, 0) # RGB for imageio/PIL, but wait, OpenCV uses BGR. imageio expects RGB.
    # Since we are drawing on OpenCV image (BGR), we should use BGR for drawing, then convert to RGB for imageio.
    draw_color = (0, 0, 255) # BGR red
    
    for i in range(num_frames):
        frame = img.copy()
        
        if i > 0:
            current_peak = (i - 1) // 3
            stage = (i - 1) % 3 + 1
            
            for p_idx, peak in enumerate(peaks):
                if p_idx < current_peak:
                    # Fully drawn
                    cv2.circle(frame, peak, dot_radius, draw_color, -1, cv2.LINE_AA)
                    cv2.circle(frame, peak, outline_radius, draw_color, 3, cv2.LINE_AA)
                elif p_idx == current_peak:
                    # Partially drawn
                    cv2.circle(frame, peak, dot_radius, draw_color, -1, cv2.LINE_AA)
                    angle = stage * 120
                    cv2.ellipse(frame, peak, (outline_radius, outline_radius), 0, 0, angle, draw_color, 3, cv2.LINE_AA)
                    
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_rgb)
        
    writer.close()
    print("Video generated successfully.")

if __name__ == "__main__":
    main()
