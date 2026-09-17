import cv2
import numpy as np
import imageio
import os

def main():
    # 1. Load image and identify the background
    img = cv2.imread('/app/first_frame.png')
    if img is None:
        raise ValueError("Could not load /app/first_frame.png")
        
    bg_color = img[0, 0].copy()

    # 2. Find shapes (connected components ignoring background)
    diff = np.any(img != bg_color, axis=-1)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(diff.astype(np.uint8))

    # 3. Get the color of each shape
    shape_colors = {}
    color_counts = {}
    
    for i in range(1, num_labels):
        # Pick a pixel in the component to get its color
        coords = np.where(labels == i)
        y, x = coords[0][0], coords[1][0]
        color = tuple(img[y, x])
        shape_colors[i] = color
        color_counts[color] = color_counts.get(color, 0) + 1

    # 4. Find the color that appears only once
    unique_color = None
    for color, count in color_counts.items():
        if count == 1:
            unique_color = color
            break

    if unique_color is None:
        raise ValueError("No unique color found")

    # 5. Find the shape with that unique color
    target_label = None
    for i, color in shape_colors.items():
        if color == unique_color:
            target_label = i
            break

    # 6. Extract the contour of the target shape
    target_mask = (labels == target_label).astype(np.uint8) * 255
    contours, _ = cv2.findContours(target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("No contour found for the target shape")
        
    target_contour = max(contours, key=cv2.contourArea)

    # 7. Generate frames and save video
    os.makedirs('/app/output', exist_ok=True)
    
    # imageio writer
    out_path = '/app/output/video.mp4'
    writer = imageio.get_writer(out_path, fps=16, codec='libx264', pixelformat='yuv420p', macro_block_size=None)
    
    total_frames = 21
    
    for i in range(total_frames):
        frame = img.copy()
        
        if i == total_frames - 1:
            cv2.drawContours(frame, [target_contour], -1, (0, 0, 0), 8, lineType=cv2.LINE_AA)
        else:
            num_points = int(len(target_contour) * i / (total_frames - 1))
            if num_points > 1:
                points_to_draw = target_contour[:num_points]
                cv2.polylines(frame, [points_to_draw], isClosed=False, color=(0, 0, 0), thickness=8, lineType=cv2.LINE_AA)
                
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_rgb)
        
    writer.close()

if __name__ == '__main__':
    main()
