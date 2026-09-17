import cv2
import numpy as np
import imageio

def make_video():
    first_frame = cv2.imread('/app/first_frame.png')
    
    # Bounding box of the 3rd shape (the Octagon)
    x_src, y_src, w_src, h_src = 391, 463, 99, 98
    src_crop = first_frame[y_src:y_src+h_src, x_src:x_src+w_src].copy()
    
    # Target region to update, which covers the question mark and the new shape position
    update_x1, update_y1 = 800, 440
    update_x2, update_y2 = 950, 580
    
    # Expected position for the 6th shape (completing the sequence)
    x_dst, y_dst = 822, 463
    
    # The final visual state for the target region: 
    # pure white background with the copied shape pasted at the correct location
    target_bg = np.ones((update_y2 - update_y1, update_x2 - update_x1, 3), dtype=np.uint8) * 255
    off_x = x_dst - update_x1
    off_y = y_dst - update_y1
    target_bg[off_y:off_y+h_src, off_x:off_x+w_src] = src_crop
    
    # The original visual state for the target region (contains the question mark)
    orig_patch = first_frame[update_y1:update_y2, update_x1:update_x2].copy()
    
    num_frames = 25
    fps = 16
    
    output_path = '/app/output/video.mp4'
    writer = imageio.get_writer(output_path, fps=fps, codec='libx264', pixelformat='yuv420p')
    
    for i in range(num_frames):
        # Calculate alpha for crossfading
        alpha = i / (num_frames - 1)
        
        # Crossfade between the original region and the final target region
        blended_patch = cv2.addWeighted(orig_patch, 1 - alpha, target_bg, alpha, 0)
        
        frame = first_frame.copy()
        frame[update_y1:update_y2, update_x1:update_x2] = blended_patch
        
        # Write to video (convert BGR to RGB for imageio)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_rgb)
        
    writer.close()

if __name__ == '__main__':
    make_video()
