import cv2
import numpy as np
import imageio

def solve():
    # Load first frame
    img = cv2.imread('/app/first_frame.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 1. Clean background by replacing agent with yellow
    purple_mask = np.all(img == [128, 0, 128], axis=-1)
    clean_bg = img.copy()
    clean_bg[purple_mask] = [255, 255, 0] # Yellow in RGB

    # 2. Extract agent offsets relative to its center
    p_y, p_x = np.where(purple_mask)
    cy, cx = p_y.mean(), p_x.mean()
    rel_y = p_y - int(cy)
    rel_x = p_x - int(cx)

    # 3. Define path nodes (shortest path from start to end)
    # Start: (0, 7), End: (6, 8)
    path_cells = [
        (0, 7),
        (1, 7),
        (2, 7),
        (3, 7),
        (4, 7),
        (5, 7),
        (6, 7),
        (6, 8)
    ]
    # Grid math: cells are 100x100 with 2px borders, so cell spacing is 102
    # Centers are at (r * 102 + 51, c * 102 + 51)
    nodes = [(r * 102 + 51, c * 102 + 51) for (r, c) in path_cells]
    
    num_frames = 55
    out_path = '/app/output/video.mp4'
    writer = imageio.get_writer(out_path, fps=16, codec='libx264', pixelformat='yuv420p', quality=10)

    for f in range(num_frames):
        frame = clean_bg.copy()
        
        # Calculate current position along the path
        progress = (f / (num_frames - 1)) * (len(nodes) - 1)
        segment_idx = int(progress)
        
        if segment_idx >= len(nodes) - 1:
            segment_idx = len(nodes) - 2
            t = 1.0
        else:
            t = progress - segment_idx
            
        y0, x0 = nodes[segment_idx]
        y1, x1 = nodes[segment_idx + 1]
        
        # Linearly interpolate position in current segment
        new_cy = y0 + t * (y1 - y0)
        new_cx = x0 + t * (x1 - x0)
        
        # Calculate destination pixel coordinates for the agent
        draw_y = np.round(new_cy).astype(int) + rel_y
        draw_x = np.round(new_cx).astype(int) + rel_x
        
        # Draw agent
        frame[draw_y, draw_x] = [128, 0, 128] # Purple in RGB
        
        # Append frame
        writer.append_data(frame)
        
    writer.close()

if __name__ == '__main__':
    solve()
