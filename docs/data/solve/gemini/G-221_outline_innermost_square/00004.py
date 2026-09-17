import cv2
import numpy as np
import imageio
import os

def get_perimeter_points(x_min, x_max, y_min, y_max):
    pts = []
    # Top edge
    for x in range(x_min, x_max + 1):
        pts.append((x, y_min))
    # Right edge
    for y in range(y_min + 1, y_max + 1):
        pts.append((x_max, y))
    # Bottom edge
    for x in range(x_max - 1, x_min - 1, -1):
        pts.append((x, y_max))
    # Left edge
    for y in range(y_max - 1, y_min, -1):
        pts.append((x_min, y))
    return pts

def find_innermost_square(img):
    # Find the exact coordinates of the innermost square
    # Since they are concentric squares, the center of the image 
    # will definitely fall inside the innermost square.
    cy, cx = img.shape[0] // 2, img.shape[1] // 2
    center_color = img[cy, cx]
    mask = np.all(img == center_color, axis=-1)
    y_indices, x_indices = np.where(mask)
    
    # If the mask covers the whole image, it might not be a solid center.
    # But for our specific image, we know it's a solid colored square.
    if len(x_indices) > 0 and len(x_indices) < img.shape[0] * img.shape[1] * 0.9:
        return int(x_indices.min()), int(x_indices.max()), int(y_indices.min()), int(y_indices.max())
    
    # Fallback to known coordinates
    return 432, 592, 432, 592

def generate_video():
    img = cv2.imread('/app/first_frame.png')
    if img is None:
        raise FileNotFoundError("Could not find /app/first_frame.png")
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    x_min, x_max, y_min, y_max = find_innermost_square(img)
    
    num_frames = 85
    fps = 16
    
    os.makedirs('/app/output', exist_ok=True)
    
    writer = imageio.get_writer(
        '/app/output/video.mp4', 
        fps=fps, 
        codec='libx264',
        pixelformat='yuv420p',
        macro_block_size=1
    )
    
    pts = get_perimeter_points(x_min, x_max, y_min, y_max)
    total_pts = len(pts)
    
    for i in range(num_frames):
        frame = img_rgb.copy()
        
        progress = i / (num_frames - 1)
        pts_to_draw = int(progress * total_pts)
        
        if pts_to_draw > 0:
            if pts_to_draw >= total_pts:
                pts_to_draw = total_pts
                current_pts = pts[:pts_to_draw]
                poly_pts = np.array(current_pts, np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [poly_pts], isClosed=True, color=(0, 0, 255), thickness=8, lineType=cv2.LINE_AA)
            else:
                current_pts = pts[:pts_to_draw]
                poly_pts = np.array(current_pts, np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [poly_pts], isClosed=False, color=(0, 0, 255), thickness=8, lineType=cv2.LINE_AA)
            
        writer.append_data(frame)
        
    writer.close()

if __name__ == '__main__':
    generate_video()
