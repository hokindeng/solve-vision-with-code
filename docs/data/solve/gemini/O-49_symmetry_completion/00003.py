import cv2
import numpy as np
import imageio

def solve():
    img = cv2.imread('/app/first_frame.png')
    
    grid_rows = [206, 309, 411, 514, 616, 718]
    grid_rows_end = [305, 407, 510, 612, 714, 817]
    grid_cols = [206, 309, 411, 514, 616, 718]
    grid_cols_end = [305, 407, 510, 612, 714, 817]
    
    cells_to_fill = [
        (0, 4),
        (2, 4),
        (2, 5),
        (5, 5)
    ]
    
    num_frames = 35
    fps = 16
    
    writer = imageio.get_writer(
        '/app/output/video.mp4', 
        fps=fps, 
        format='FFMPEG',
        codec='libx264',
        pixelformat='yuv420p',
        macro_block_size=1
    )
    
    for f in range(num_frames):
        frame_img = img.copy()
        
        for i, (r, c) in enumerate(cells_to_fill):
            start_frame = 2 + i * 7
            end_frame = start_frame + 7
            
            if f >= end_frame:
                progress = 1.0
            elif f <= start_frame:
                progress = 0.0
            else:
                progress = (f - start_frame) / (end_frame - start_frame)
            
            y_start = grid_rows[r]
            y_end = grid_rows_end[r]
            x_start = grid_cols[c]
            x_end = grid_cols_end[c]
            
            if progress >= 1.0:
                frame_img[y_start:y_end+1, x_start:x_end+1] = [237, 58, 124]
            elif progress > 0:
                cy = (y_start + y_end) / 2.0
                cx = (x_start + x_end) / 2.0
                
                h = (y_end - y_start + 1) * progress
                w = (x_end - x_start + 1) * progress
                
                y0 = int(round(cy - h / 2.0))
                y1 = int(round(cy + h / 2.0))
                x0 = int(round(cx - w / 2.0))
                x1 = int(round(cx + w / 2.0))
                
                y0 = max(y_start, y0)
                y1 = min(y_end + 1, y1)
                x0 = max(x_start, x0)
                x1 = min(x_end + 1, x1)
                
                if y1 > y0 and x1 > x0:
                    frame_img[y0:y1, x0:x1] = [237, 58, 124]
                
        # convert BGR to RGB for imageio
        frame_rgb = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_rgb)
        
    writer.close()

if __name__ == '__main__':
    solve()
