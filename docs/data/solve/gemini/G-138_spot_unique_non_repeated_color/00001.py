import cv2
import numpy as np
import os
import subprocess
from collections import Counter

def main():
    # 1. Read the first frame
    img = cv2.imread('/app/first_frame.png')
    if img is None:
        raise FileNotFoundError("Could not read /app/first_frame.png")
        
    # 2. Identify the background and the shapes
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Background is white (255), so threshold at 250 to get a mask of the shapes
    _, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)

    # Extract connected components (shapes)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    
    shape_colors = []
    # Collect colors of all shapes (exclude background label 0)
    for i in range(1, num_labels):
        mask = (labels == i)
        # Extract the color of the shape using the first pixel's value in the mask
        color = tuple(img[mask][0])
        shape_colors.append((i, color))
        
    # 3. Find the shape with the unique color
    color_counts = Counter(color for _, color in shape_colors)
    
    unique_color = None
    for color, count in color_counts.items():
        if count == 1:
            unique_color = color
            break
            
    if unique_color is None:
        raise ValueError("No unique color found among shapes.")
        
    # 4. Find the mask for the unique colored shape
    target_mask = None
    for i, color in shape_colors:
        if color == unique_color:
            target_mask = (labels == i).astype(np.uint8)
            break
            
    # 5. Get its external contour
    # cv2.RETR_EXTERNAL ensures we only get the outer boundary
    contours, _ = cv2.findContours(target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("No contour found for the target shape.")
        
    contour = contours[0]
    
    # 6. Generate the video showing the drawing process
    num_frames = 21
    fps = 16
    os.makedirs('/app/output', exist_ok=True)
    out_path = '/app/output/video.mp4'
    
    # Setup ffmpeg subprocess for raw video input
    process = subprocess.Popen([
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{img.shape[1]}x{img.shape[0]}', '-pix_fmt', 'bgr24', '-r', str(fps),
        '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        out_path
    ], stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    
    for i in range(num_frames):
        frame = img.copy()
        
        if i > 0:
            # Calculate how many points to draw (paced progressively from 0 to total points)
            pts_to_draw = int(len(contour) * i / (num_frames - 1))
            
            if pts_to_draw > 0:
                is_closed = (i == num_frames - 1)
                # Draw the contour outline progressively
                cv2.polylines(frame, [contour[:pts_to_draw]], isClosed=is_closed, 
                              color=(0, 0, 0), thickness=4, lineType=cv2.LINE_AA)
                
        # Write the frame to the ffmpeg process
        process.stdin.write(frame.tobytes())
        
    process.stdin.close()
    process.wait()

if __name__ == '__main__':
    main()
