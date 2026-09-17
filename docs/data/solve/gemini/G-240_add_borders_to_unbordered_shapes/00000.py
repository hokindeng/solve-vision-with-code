import os
import cv2
import numpy as np
import imageio

def make_video():
    os.makedirs('/app/output', exist_ok=True)
    
    # Load first frame
    img = cv2.imread('/app/first_frame.png')
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Background is white. Let's find non-white components
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY_INV)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    
    shapes_to_modify = []
    for i in range(1, num_labels):
        mask = (labels == i)
        colors = np.unique(img_rgb[mask], axis=0)
        has_black = any((c == [0, 0, 0]).all() for c in colors)
        if not has_black:
            shapes_to_modify.append(i)
            
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    
    border_info = []
    for i in shapes_to_modify:
        comp = (labels == i).astype(np.uint8)
        cy, cx = centroids[i][1], centroids[i][0]
        
        eroded = cv2.erode(comp, kernel)
        border_mask = (comp - eroded) > 0
        
        y, x = np.where(border_mask)
        dy = y - cy
        dx = x - cx
        angle = np.arctan2(dy, dx)
        clock_angle = (angle + np.pi / 2) % (2 * np.pi)
        
        border_info.append({
            'y': y,
            'x': x,
            'clock_angle': clock_angle
        })
        
    fps = 16
    total_frames = 80
    
    writer = imageio.get_writer('/app/output/video.mp4', fps=fps, codec='libx264', pixelformat='yuv420p', macro_block_size=None)
    
    for f in range(total_frames):
        frame_img = img_rgb.copy()
        
        # progress goes from 0 to 1
        progress = f / (total_frames - 1)
        threshold_angle = progress * 2 * np.pi
        
        if f == 0:
            pass
        elif f == total_frames - 1:
            for info in border_info:
                y = info['y']
                x = info['x']
                frame_img[y, x] = [0, 0, 0]
        else:
            for info in border_info:
                y = info['y']
                x = info['x']
                clock_angle = info['clock_angle']
                
                draw_mask = clock_angle <= threshold_angle
                frame_img[y[draw_mask], x[draw_mask]] = [0, 0, 0]
            
        writer.append_data(frame_img)
        
    writer.close()

if __name__ == '__main__':
    make_video()
