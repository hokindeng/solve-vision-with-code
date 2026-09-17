import cv2
import numpy as np
import os
import imageio

def main():
    os.makedirs('/app/output', exist_ok=True)
    img = cv2.imread('/app/first_frame.png')
    
    # Background is white. Find foreground mask
    bg_color = np.array([255, 255, 255])
    fg_mask = np.any(img != bg_color, axis=-1).astype(np.uint8) * 255
    
    # Get connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(fg_mask, connectivity=8)
    
    # Find the color of each component
    comp_colors = {}
    for i in range(1, num_labels):
        # We can just take the color at the centroid, or the most frequent color
        comp_mask = (labels == i)
        pixels = img[comp_mask]
        # Find unique colors in this component
        colors, counts = np.unique(pixels, axis=0, return_counts=True)
        # Take the most frequent color
        main_color = colors[np.argmax(counts)]
        comp_colors[i] = tuple(main_color)
        
    # Count occurrences of each color among components
    color_counts = {}
    for color in comp_colors.values():
        color_counts[color] = color_counts.get(color, 0) + 1
        
    # Find the target color (appears exactly once)
    target_color = None
    for color, count in color_counts.items():
        if count == 1:
            target_color = color
            break
            
    if target_color is None:
        raise ValueError("No unique color found")
        
    # Find the component with this target color
    target_comp_label = None
    for label, color in comp_colors.items():
        if color == target_color:
            target_comp_label = label
            break
            
    # Create mask for this component
    target_mask = (labels == target_comp_label).astype(np.uint8) * 255
    
    # Find contour
    contours, _ = cv2.findContours(target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # Get the largest contour in case of noise
    cnt = max(contours, key=cv2.contourArea)
    
    n_frames = 21
    fps = 16
    
    writer = imageio.get_writer('/app/output/video.mp4', fps=fps, codec='libx264', pixelformat='yuv420p')
    
    total_points = len(cnt)
    
    for i in range(n_frames):
        frame = img.copy()
        
        # Calculate how many points to draw in this frame
        if i == 0:
            pts_to_draw = 0
        elif i == n_frames - 1:
            pts_to_draw = total_points
        else:
            pts_to_draw = int((i / (n_frames - 1)) * total_points)
        
        if pts_to_draw > 1:
            pts = cnt[:pts_to_draw]
            # Draw lines
            cv2.polylines(frame, [pts], isClosed=False, color=(0, 0, 0), thickness=6, lineType=cv2.LINE_AA)
            
            # If it's the last frame, ensure it's closed
            if i == n_frames - 1:
                cv2.polylines(frame, [cnt], isClosed=True, color=(0, 0, 0), thickness=6, lineType=cv2.LINE_AA)
                
        # Convert BGR to RGB for imageio
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_rgb)
        
    writer.close()
    print("Video generated at /app/output/video.mp4")

if __name__ == "__main__":
    main()
