import cv2
import numpy as np
import imageio
import os

def main():
    input_path = '/app/first_frame.png'
    output_path = '/app/output/video.mp4'
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    img = cv2.imread(input_path)
    # OpenCV loads in BGR, imageio expects RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    bg_color = np.array([255, 255, 255], dtype=np.uint8)
    diff = np.any(img_rgb != bg_color, axis=-1)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(diff.astype(np.uint8))
    
    objects = []
    base_img = img_rgb.copy()
    
    for i in range(1, num_labels):
        # We assume large connected components (area > 1000) are the 2D objects to rotate
        if stats[i, cv2.CC_STAT_AREA] > 1000:
            mask = (labels == i)
            objects.append({
                'mask': mask,
                'centroid': tuple(centroids[i])
            })
            base_img[mask] = bg_color
            
    num_frames = 17
    fps = 16
    total_angle = 176.0
    
    writer = imageio.get_writer(output_path, fps=fps, codec='libx264', pixelformat='yuv420p')
    
    for i in range(num_frames):
        angle = total_angle * i / (num_frames - 1)
        
        frame = base_img.copy()
        
        for obj in objects:
            centroid = obj['centroid']
            mask = obj['mask']
            
            # Create object image with white background
            obj_img = np.full_like(img_rgb, 255)
            obj_img[mask] = img_rgb[mask]
            
            # Create object mask
            obj_mask = np.zeros(img_rgb.shape[:2], dtype=np.uint8)
            obj_mask[mask] = 255
            
            # Get rotation matrix
            M = cv2.getRotationMatrix2D(centroid, angle, 1.0)
            
            # Rotate object image and mask
            # Using INTER_NEAREST to preserve exact colors and sharp edges
            rot_obj = cv2.warpAffine(obj_img, M, (img_rgb.shape[1], img_rgb.shape[0]), 
                                     flags=cv2.INTER_NEAREST, 
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
            
            rot_mask = cv2.warpAffine(obj_mask, M, (img_rgb.shape[1], img_rgb.shape[0]), 
                                      flags=cv2.INTER_NEAREST, 
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            
            # Paste the rotated object onto the frame
            frame[rot_mask > 0] = rot_obj[rot_mask > 0]
            
        writer.append_data(frame)
        
    writer.close()

if __name__ == '__main__':
    main()
