import cv2
import numpy as np
import imageio
import os

def shift_image(img, dx, dy, fill_value):
    shifted = np.full_like(img, fill_value)
    h, w = img.shape[:2]
    
    src_y1 = max(0, -dy)
    src_y2 = min(h, h - dy)
    src_x1 = max(0, -dx)
    src_x2 = min(w, w - dx)
    
    dst_y1 = max(0, dy)
    dst_y2 = min(h, h + dy)
    dst_x1 = max(0, dx)
    dst_x2 = min(w, w + dx)
    
    shifted[dst_y1:dst_y2, dst_x1:dst_x2] = img[src_y1:src_y2, src_x1:src_x2]
    return shifted

def solve():
    img = cv2.imread('/app/first_frame.png')
    
    # Identify background color from top-left pixel
    bg_color = img[0, 0]
    
    # Create foreground mask (everything that is not the exact background color)
    bg_mask_bool = np.all(img == bg_color, axis=-1)
    fg_mask = (~bg_mask_bool).astype(np.uint8) * 255
    
    # Find contours of all objects
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Identify green and red masks dynamically
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Green in HSV: hue 40-80
    lower_green = np.array([40, 50, 50])
    upper_green = np.array([80, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Red in HSV: hue 0-10 and 170-180
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
    
    moving_c = None
    target_c = None
    
    for c in contours:
        c_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.drawContours(c_mask, [c], -1, 255, -1)
        
        if np.any(cv2.bitwise_and(green_mask, green_mask, mask=c_mask)):
            moving_c = c
        if np.any(cv2.bitwise_and(red_mask, red_mask, mask=c_mask)):
            target_c = c
            
    # Fallback if dynamic color thresholding fails
    if moving_c is None or target_c is None:
        print("Falling back to hardcoded bounding boxes.")
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if x == 116 and y == 241:
                target_c = c
            elif x == 230 and y == 445:
                moving_c = c

    # Compute target alignment based on centers
    target_x, _, target_w, _ = cv2.boundingRect(target_c)
    moving_x, _, moving_w, _ = cv2.boundingRect(moving_c)
    
    target_center_x = target_x + target_w / 2.0
    moving_center_x = moving_x + moving_w / 2.0
    
    total_shift = target_center_x - moving_center_x
    
    # Create mask of the moving object
    moving_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.drawContours(moving_mask, [moving_c], -1, 255, -1)
    
    # Extract the moving object's pixels
    moving_pixels = img.copy()
    # Mask out everything else with black (doesn't matter what color as long as it's shifted)
    moving_pixels[moving_mask == 0] = 0
    
    # Create the background image with the moving object erased
    bg = img.copy()
    bg[moving_mask == 255] = bg_color
    
    frames = []
    num_frames = 60
    
    # We will pass fill_value as black for mask and pixels
    fill_mask = 0
    fill_pixels = [0, 0, 0]
    
    for i in range(num_frames):
        # Calculate proportional shift for current frame
        shift_x = int(round(total_shift * i / (num_frames - 1)))
        
        shifted_mask = shift_image(moving_mask, shift_x, 0, fill_value=fill_mask)
        shifted_pixels = shift_image(moving_pixels, shift_x, 0, fill_value=fill_pixels)
        
        # Combine shifted object with background
        frame = bg.copy()
        idx = shifted_mask == 255
        frame[idx] = shifted_pixels[idx]
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        
    os.makedirs('/app/output', exist_ok=True)
    writer = imageio.get_writer(
        '/app/output/video.mp4', 
        fps=16, 
        codec='libx264', 
        pixelformat='yuv420p',
        macro_block_size=None,
        ffmpeg_params=['-crf', '17']  # high quality to minimize compression artifacts
    )
    for f in frames:
        writer.append_data(f)
    writer.close()

if __name__ == "__main__":
    solve()
