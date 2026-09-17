import cv2
import numpy as np
import imageio
import os

def main():
    img_bgr = cv2.imread('/app/first_frame.png')
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    bg_color = np.array([245, 250, 255])
    
    mask_fg = (~np.all(img_rgb == bg_color, axis=-1)).astype(np.uint8)
    
    mask_left = mask_fg.copy()
    mask_left[:, 512:] = 0
    
    mask_right = mask_fg.copy()
    mask_right[:, :512] = 0
    
    num_labels_L, labels_L, stats_L, centroids_L = cv2.connectedComponentsWithStats(mask_left, connectivity=8)
    num_labels_R, labels_R, stats_R, centroids_R = cv2.connectedComponentsWithStats(mask_right, connectivity=8)
    
    faces_info = []
    for i in range(1, num_labels_L):
        x, y, w, h, area = stats_L[i]
        if area > 100:
            faces_info.append({'id': i, 'bbox': (x,y,w,h)})
            
    outlines = []
    for i in range(1, num_labels_R):
        x, y, w, h, area = stats_R[i]
        if area > 100 and w > 10 and h > 10:
            outlines.append({'id': i, 'bbox': (x,y,w,h)})
            
    # Calculate background (remove faces)
    bg_img = img_rgb.copy()
    faces_data = []
    
    for f in faces_info:
        x1, y1, w1, h1 = f['bbox']
        best_diff = 1e9
        best_dx, best_dy = 0, 0
        
        for o in outlines:
            x2, y2, w2, h2 = o['bbox']
            diff = abs(w1 - w2) + abs(h1 - h2)
            if diff < best_diff:
                best_diff = diff
                # BBox center difference
                cx1 = x1 + w1 / 2.0
                cy1 = y1 + h1 / 2.0
                cx2 = x2 + w2 / 2.0
                cy2 = y2 + h2 / 2.0
                best_dx = cx2 - cx1
                best_dy = cy2 - cy1
                
        # Remove face from background
        mask = (labels_L == f['id'])
        bg_img[mask] = bg_color
        
        y_idx, x_idx = np.where(mask)
        colors = img_rgb[y_idx, x_idx]
        
        faces_data.append({
            'y': y_idx,
            'x': x_idx,
            'colors': colors,
            'dx': best_dx,
            'dy': best_dy
        })
        
    os.makedirs('/app/output', exist_ok=True)
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', pixelformat='yuv420p', macro_block_size=None)
    
    num_frames = 64
    for f_idx in range(num_frames):
        progress = f_idx / (num_frames - 1)
        
        frame = bg_img.copy()
        
        for face in faces_data:
            cur_dx = int(round(face['dx'] * progress))
            cur_dy = int(round(face['dy'] * progress))
            
            new_y = face['y'] + cur_dy
            new_x = face['x'] + cur_dx
            
            frame[new_y, new_x] = face['colors']
            
        writer.append_data(frame)
        
    writer.close()

if __name__ == '__main__':
    main()
