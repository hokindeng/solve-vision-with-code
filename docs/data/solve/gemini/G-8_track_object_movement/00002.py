import cv2
import numpy as np
import imageio

def solve():
    # Read the first frame
    img = cv2.imread('/app/first_frame.png')
    if img is None:
        raise ValueError("Could not read /app/first_frame.png")
    
    # imageio uses RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 1. Dynamically find the background color (most frequent color)
    unique, counts = np.unique(img_rgb.reshape(-1, 3), axis=0, return_counts=True)
    bg_color = unique[np.argmax(counts)]

    # 2. Find the green border (G > 150, R < 50, B < 50)
    green_mask = ((img_rgb[:,:,1] > 150) & (img_rgb[:,:,0] < 50) & (img_rgb[:,:,2] < 50)).astype(np.uint8) * 255
    contours_green, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c_green = max(contours_green, key=cv2.contourArea)
    x_g, y_g, w_g, h_g = cv2.boundingRect(c_green)

    # 3. Find the moving object (the entire non-bg object containing the green border)
    non_bg_mask = np.any(img_rgb != bg_color, axis=-1).astype(np.uint8) * 255
    contours_all, _ = cv2.findContours(non_bg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    moving_bbox = None
    c_moving = None
    for c in contours_all:
        x, y, w, h = cv2.boundingRect(c)
        # Check if the green border is inside this object's bounding box
        if x <= x_g + w_g/2 <= x+w and y <= y_g + h_g/2 <= y+h:
            moving_bbox = (x, y, w, h)
            c_moving = c
            break
            
    if moving_bbox is None:
        raise ValueError("Could not find the moving object.")
        
    x, y, w, h = moving_bbox

    # 4. Find the red star (R > 150, G < 50, B < 50)
    red_mask = ((img_rgb[:,:,0] > 150) & (img_rgb[:,:,1] < 50) & (img_rgb[:,:,2] < 50)).astype(np.uint8) * 255
    contours_red, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c_red = max(contours_red, key=cv2.contourArea)

    # 5. Calculate centers and the total horizontal shift required
    M_red = cv2.moments(c_red)
    cx_target = int(M_red["m10"] / M_red["m00"])

    M_moving = cv2.moments(c_moving)
    cx_moving = int(M_moving["m10"] / M_moving["m00"])

    total_shift = cx_target - cx_moving

    # 6. Generate the video
    num_frames = 60
    fps = 16

    # Extract the original object and its non-background mask
    object_crop = img_rgb[y:y+h, x:x+w].copy()
    mask = np.any(object_crop != bg_color, axis=-1)

    # Open video writer
    writer = imageio.get_writer('/app/output/video.mp4', fps=fps, codec='libx264', pixelformat='yuv420p')

    for i in range(num_frames):
        # Calculate the shift for this specific frame
        shift = round(total_shift * i / (num_frames - 1))
        
        new_img = img_rgb.copy()
        
        # Erase the object from its original position
        new_img[y:y+h, x:x+w][mask] = bg_color
        
        # Draw the object at its new position
        new_x = x + shift
        new_img[y:y+h, new_x:new_x+w][mask] = object_crop[mask]
        
        # Append frame to video
        writer.append_data(new_img)

    writer.close()

if __name__ == '__main__':
    solve()
