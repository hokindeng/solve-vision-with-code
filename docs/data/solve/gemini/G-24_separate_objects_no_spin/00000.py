import cv2
import numpy as np
import imageio
import os

def create_video():
    img = cv2.imread('/app/first_frame.png')
    
    # Isolate the background and the objects
    bg_img = img.copy()
    border_mask = cv2.inRange(img, np.array([120, 120, 120]), np.array([120, 120, 120]))
    
    colors = {
        'obj1': np.array([220, 140, 200]),
        'obj2': np.array([80, 80, 255]),
        'obj3': np.array([255, 150, 100])
    }
    
    objects = []
    
    for name, color in colors.items():
        color_mask = cv2.inRange(img, color, color)
        kernel = np.ones((15, 15), np.uint8)
        dilated = cv2.dilate(color_mask, kernel, iterations=1)
        obj_full_mask = np.logical_or(color_mask > 0, np.logical_and(border_mask > 0, dilated > 0)).astype(np.uint8)
        
        y_idx, x_idx = np.where(obj_full_mask > 0)
        min_x, max_x = x_idx.min(), x_idx.max()
        min_y, max_y = y_idx.min(), y_idx.max()
        
        obj_pixels = img[min_y:max_y+1, min_x:max_x+1].copy()
        obj_mask = obj_full_mask[min_y:max_y+1, min_x:max_x+1].copy()
        
        objects.append({
            'name': name,
            'min_x': min_x, 'max_x': max_x,
            'min_y': min_y, 'max_y': max_y,
            'pixels': obj_pixels,
            'mask': obj_mask
        })
        
        # erase from background (replace with white background color)
        bg_img[obj_full_mask > 0] = [255, 255, 255]
        
    shifts = {'obj1': 503, 'obj2': 273, 'obj3': 464}
    
    os.makedirs('/app/output', exist_ok=True)
    
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', macro_block_size=None, pixelformat='yuv420p')
    
    num_frames = 30
    for i in range(num_frames):
        frame = bg_img.copy()
        
        # Calculate current shift fraction
        t = i / (num_frames - 1)
        
        for obj in objects:
            name = obj['name']
            target_shift = shifts[name]
            current_shift = int(round(target_shift * t))
            
            min_x = obj['min_x']
            max_x = obj['max_x']
            min_y = obj['min_y']
            max_y = obj['max_y']
            mask = obj['mask']
            pixels = obj['pixels']
            
            mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
            
            frame_region = frame[min_y:max_y+1, min_x+current_shift:max_x+1+current_shift]
            
            frame[min_y:max_y+1, min_x+current_shift:max_x+1+current_shift] = np.where(
                mask_3d > 0,
                pixels,
                frame_region
            )
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_rgb)
        
    writer.close()

if __name__ == '__main__':
    create_video()
