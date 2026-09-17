import cv2
import numpy as np
import os
import subprocess

def create_video():
    os.makedirs('/app/output', exist_ok=True)

    img = cv2.imread('/app/first_frame.png')
    if img is None:
        raise ValueError("Could not read /app/first_frame.png")
        
    green_mask = np.all(img == [70, 140, 70], axis=-1)

    # Base image without the green box
    base_img = img.copy()
    base_img[green_mask] = [255, 255, 255]

    # Get the bounding box of the green box
    y_indices, x_indices = np.where(green_mask)
    min_y, max_y = np.min(y_indices), np.max(y_indices)
    min_x, max_x = np.min(x_indices), np.max(x_indices)

    box_h = max_y - min_y + 1
    box_w = max_x - min_x + 1

    box_mask = green_mask[min_y:max_y+1, min_x:max_x+1]

    # Find the centers of the left and right objects
    # Background is [255, 255, 255]
    obj_mask = np.any(img != [255, 255, 255], axis=-1) & ~green_mask
    
    contours, _ = cv2.findContours(obj_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by x coordinate
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
    
    if len(contours) >= 2:
        left_obj = cv2.boundingRect(contours[0])
        right_obj = cv2.boundingRect(contours[-1])
        
        left_cx = left_obj[0] + left_obj[2] / 2.0
        right_cx = right_obj[0] + right_obj[2] / 2.0
        
        left_cy = left_obj[1] + left_obj[3] / 2.0
        right_cy = right_obj[1] + right_obj[3] / 2.0
        
        shift_x = right_cx - left_cx
        shift_y = right_cy - left_cy
    else:
        shift_x = 502.5
        shift_y = 0.0

    start_x = float(min_x)
    end_x = start_x + shift_x
    start_y = float(min_y)
    end_y = start_y + shift_y

    num_frames = 25
    fps = 16

    height, width = img.shape[:2]

    ffmpeg_cmd = [
        'ffmpeg',
        '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{width}x{height}',
        '-pix_fmt', 'bgr24',
        '-r', str(fps),
        '-i', '-',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '/app/output/video.mp4'
    ]

    process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    for i in range(num_frames):
        t = i / (num_frames - 1)
        
        curr_x = int(round(start_x + t * (end_x - start_x)))
        curr_y = int(round(start_y + t * (end_y - start_y)))
        
        frame = base_img.copy()
        
        # Ensure we don't go out of bounds
        y_start = max(0, curr_y)
        y_end = min(height, curr_y + box_h)
        x_start = max(0, curr_x)
        x_end = min(width, curr_x + box_w)
        
        mask_y_start = y_start - curr_y
        mask_y_end = box_h - (curr_y + box_h - y_end)
        mask_x_start = x_start - curr_x
        mask_x_end = box_w - (curr_x + box_w - x_end)
        
        region = frame[y_start:y_end, x_start:x_end]
        mask_region = box_mask[mask_y_start:mask_y_end, mask_x_start:mask_x_end]
        
        region[mask_region] = [70, 140, 70]
        
        process.stdin.write(frame.tobytes())

    process.stdin.close()
    process.wait()

if __name__ == '__main__':
    create_video()
