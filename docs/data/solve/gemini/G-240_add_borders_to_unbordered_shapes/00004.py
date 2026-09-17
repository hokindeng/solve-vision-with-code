import cv2
import numpy as np
import imageio
import os

def solve():
    img = cv2.imread('/app/first_frame.png')
    
    bg_mask = cv2.inRange(img, (255, 255, 255), (255, 255, 255))
    black_mask = cv2.inRange(img, (0, 0, 0), (0, 0, 0))

    shape_pixels_mask = cv2.bitwise_not(cv2.bitwise_or(bg_mask, black_mask))
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(shape_pixels_mask, connectivity=8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

    shapes_info = []

    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < 100:
            continue
            
        mask = (labels == i).astype(np.uint8) * 255
        target_border = cv2.bitwise_and(cv2.dilate(mask, kernel), cv2.bitwise_not(mask))
        
        intersect = cv2.bitwise_and(target_border, black_mask)
        intersect_count = np.count_nonzero(intersect)
        
        if intersect_count < 100:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if contours:
                c = max(contours, key=cv2.contourArea)
                shapes_info.append({
                    'target_border': target_border,
                    'contour': c
                })

    num_frames = 80
    fps = 16
    
    os.makedirs('/app/output', exist_ok=True)
    writer = imageio.get_writer('/app/output/video.mp4', fps=fps, codec='libx264', pixelformat='yuv420p', macro_block_size=None)
    
    for f in range(num_frames):
        progress = f / max(1, (num_frames - 1))
        
        frame = img.copy()
        
        for info in shapes_info:
            c = info['contour']
            target_border = info['target_border']
            
            n_points = int(progress * len(c))
            
            reveal_mask = np.zeros(img.shape[:2], dtype=np.uint8)
            if n_points > 0:
                pts = c[:n_points, 0, :]
                for pt in pts:
                    cv2.circle(reveal_mask, tuple(pt), 15, 255, -1)
            
            current_border = cv2.bitwise_and(target_border, reveal_mask)
            
            frame[current_border > 0] = (0, 0, 0)
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_rgb)
        
    writer.close()

if __name__ == '__main__':
    solve()
