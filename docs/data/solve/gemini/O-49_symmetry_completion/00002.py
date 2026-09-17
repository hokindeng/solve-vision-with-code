import cv2
import numpy as np
import subprocess
import os
import tempfile

def solve():
    img = cv2.imread('/app/first_frame.png')
    if img is None:
        raise FileNotFoundError("Could not read /app/first_frame.png")
        
    grid_color = np.array([225, 213, 203])
    mask = np.all(img == grid_color, axis=-1)
    row_counts = np.sum(mask, axis=1)
    col_counts = np.sum(mask, axis=0)

    rows = np.where(row_counts > 100)[0]
    cols = np.where(col_counts > 100)[0]

    def group_lines(lines):
        groups = []
        current_group = []
        for l in lines:
            if not current_group:
                current_group.append(l)
            elif l == current_group[-1] + 1:
                current_group.append(l)
            else:
                groups.append(int(np.mean(current_group)))
                current_group = [l]
        if current_group:
            groups.append(int(np.mean(current_group)))
        return groups

    grid_y = group_lines(rows)
    grid_x = group_lines(cols)

    centers_y = [(grid_y[i] + grid_y[i+1])//2 for i in range(len(grid_y)-1)]
    centers_x = [(grid_x[i] + grid_x[i+1])//2 for i in range(len(grid_x)-1)]

    blue = np.array([235, 99, 37])
    white = np.array([255, 255, 255])

    missing_cells = []
    for r, cy in enumerate(centers_y):
        is_filled_row = False
        for c in range(4):
            cx = centers_x[c]
            if np.all(img[cy, cx] == blue):
                is_filled_row = True
                break
                
        if is_filled_row:
            for c in range(4, 8):
                cx = centers_x[c]
                if np.all(img[cy, cx] == white):
                    missing_cells.append((r, c))

    def get_bounds(cx, cy, image):
        x1, x2, y1, y2 = cx, cx, cy, cy
        while image[cy, x1, 0] != 225: x1 -= 1
        while image[cy, x2, 0] != 225: x2 += 1
        while image[y1, cx, 0] != 225: y1 -= 1
        while image[y2, cx, 0] != 225: y2 += 1
        return x1+1, x2, y1+1, y2

    cell_bounds = []
    for r, c in missing_cells:
        bounds = get_bounds(centers_x[c], centers_y[r], img)
        cell_bounds.append(bounds)
        
    if not cell_bounds:
        print("No missing cells found!")
        return

    min_x = min([b[0] for b in cell_bounds])
    max_x = max([b[1] for b in cell_bounds])
    
    os.makedirs('/app/output', exist_ok=True)
    
    num_frames = 35
    with tempfile.TemporaryDirectory() as temp_dir:
        for f in range(num_frames):
            frame_img = img.copy()
            sweep_x = int(min_x + (max_x - min_x) * (f / (num_frames - 1)))
            
            for (x1, x2, y1, y2) in cell_bounds:
                for y in range(y1, y2):
                    for x in range(x1, x2):
                        if x < sweep_x:
                            frame_img[y, x] = blue
                            
            cv2.imwrite(f'{temp_dir}/frame_{f:04d}.png', frame_img)
            
        cmd = [
            'ffmpeg', '-y', '-framerate', '16', '-i', f'{temp_dir}/frame_%04d.png',
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '/app/output/video.mp4'
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("Video generated at /app/output/video.mp4")

if __name__ == '__main__':
    solve()
