import cv2
import numpy as np
import os
import imageio

def main():
    # Load the first frame
    img = cv2.imread('/app/first_frame.png')
    # BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    bg_color = img[0, 0].copy()
    
    # Create mask for objects
    mask = np.any(img != bg_color, axis=-1).astype(np.uint8) * 255
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    # Identify cones (triangles)
    cone_mask = np.zeros_like(mask, dtype=bool)
    
    for i in range(1, num_labels):
        obj_mask = (labels == i).astype(np.uint8) * 255
        contours, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        
        # Cones are triangles, so 3 vertices
        if len(approx) == 3:
            cone_mask = cone_mask | (labels == i)
            
    # Generate frames
    num_frames = 96
    fps = 16
    
    os.makedirs('/app/output', exist_ok=True)
    # Use quality=10 for better imageio ffmpeg output
    writer = imageio.get_writer('/app/output/video.mp4', fps=fps, codec='libx264', format='FFMPEG', pixelformat='yuv420p', quality=10)
    
    for frame_idx in range(num_frames):
        alpha = 1.0 - (frame_idx / (num_frames - 1))
        alpha = max(0.0, min(1.0, alpha))
        
        frame = img.copy()
        
        # Apply fade out to cone objects
        cone_pixels = frame[cone_mask]
        faded_pixels = cone_pixels * alpha + bg_color.astype(np.float32) * (1 - alpha)
        frame[cone_mask] = faded_pixels.astype(np.uint8)
        
        writer.append_data(frame)
        
    writer.close()
    print("Video generated at /app/output/video.mp4")

if __name__ == "__main__":
    main()
