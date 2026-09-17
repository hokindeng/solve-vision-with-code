import cv2
import numpy as np
import imageio
import os

def main():
    img = cv2.imread('/app/first_frame.png')
    if img is None:
        raise ValueError("Could not read /app/first_frame.png")
        
    bg_color = np.array([255, 255, 255])
    labels_mask = np.any(img != bg_color, axis=-1).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(labels_mask, 8)
    
    shapes_to_border = []
    for comp_idx in range(1, num_labels):
        comp_mask = (labels == comp_idx).astype(np.uint8)
        
        # Check if shape already has black pixels
        comp_pixels = img[labels == comp_idx]
        has_black = np.any(np.all(comp_pixels == [0, 0, 0], axis=-1))
        
        if not has_black:
            contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if len(contours) > 0:
                pts = contours[0].reshape(-1, 2)
                shapes_to_border.append({
                    'mask': comp_mask == 1,
                    'pts': pts
                })
                
    num_frames = 80
    fps = 16
    os.makedirs('/app/output', exist_ok=True)
    out_path = '/app/output/video.mp4'
    writer = imageio.get_writer(out_path, fps=fps, codec='libx264', pixelformat='yuv420p', macro_block_size=None)
    
    for t in range(num_frames):
        frame = img.copy()
        border_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        
        for shape in shapes_to_border:
            pts = shape['pts']
            if t == num_frames - 1:
                k = len(pts)
                is_closed = True
            else:
                k = int((t / (num_frames - 1)) * len(pts))
                is_closed = False
                
            if k > 0:
                cv2.polylines(border_mask, [pts[:k]], is_closed, 255, thickness=10, lineType=cv2.LINE_8)
                
        # Apply the border, restricting it to only overwrite the original shape's pixels
        for shape in shapes_to_border:
            condition = (border_mask == 255) & shape['mask']
            frame[condition] = [0, 0, 0]
            
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(rgb_frame)
        
    writer.close()

if __name__ == '__main__':
    main()
