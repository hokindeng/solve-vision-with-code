import cv2
import numpy as np
import collections
import imageio
import os

def solve():
    os.makedirs('/app/output', exist_ok=True)
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', pixelformat='yuv420p', quality=10, macro_block_size=None)

    img = cv2.imread('/app/first_frame.png')
    # Convert BGR to RGB for imageio
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    bg_color = img[0, 0]
    mask = ~np.all(img == bg_color, axis=-1)
    ret, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)

    # Find the shape with a unique color
    color_counts = collections.defaultdict(int)
    for i in range(1, ret):
        shape_mask = (labels == i)
        color = tuple(img[shape_mask][0])
        color_counts[color] += 1

    unique_color = None
    for color, count in color_counts.items():
        if count == 1:
            unique_color = color
            break

    target_label = None
    for i in range(1, ret):
        shape_mask = (labels == i)
        if tuple(img[shape_mask][0]) == unique_color:
            target_label = i
            break

    # Get the contour for the target shape
    shape_mask = (labels == target_label).astype(np.uint8)
    contours, _ = cv2.findContours(shape_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = contours[0]
    
    # Extend the contour by 1 point to close the loop smoothly when drawing
    extended_contour = np.vstack([contour, contour[0:1]])

    num_frames = 21
    num_points = len(contour)

    for i in range(num_frames):
        frame = img_rgb.copy()
        if i > 0:
            # Calculate how many points to include based on progress
            pts_to_draw = int(round(i * num_points / (num_frames - 1))) + 1
            if pts_to_draw > len(extended_contour):
                pts_to_draw = len(extended_contour)
            
            pts = extended_contour[:pts_to_draw]
            if len(pts) >= 2:
                # Draw the progressive contour in black
                cv2.polylines(frame, [pts], isClosed=False, color=(0, 0, 0), thickness=5)
                
        writer.append_data(frame)

    writer.close()

if __name__ == '__main__':
    solve()
