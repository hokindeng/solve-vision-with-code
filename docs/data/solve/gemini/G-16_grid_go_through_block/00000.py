import cv2
import numpy as np
import imageio

def generate_path():
    path = [(9, 4)]
    def move_to(tr, tc):
        cr, cc = path[-1]
        while cc != tc:
            cc += 1 if tc > cc else -1
            path.append((cr, cc))
        while cr != tr:
            cr += 1 if tr > cr else -1
            path.append((cr, cc))
    move_to(7, 1)
    move_to(3, 6)
    move_to(7, 8)
    move_to(9, 7)
    return path

def main():
    img = cv2.imread('/app/first_frame.png')
    
    # 1. Extract agent mask and pixels
    cell = img[886:977, 421:512]
    agent_mask = ((cell == [0, 165, 255]).all(axis=-1) | (cell == [0, 120, 200]).all(axis=-1))
    
    y_min, y_max = 21, 67
    x_min, x_max = 21, 67
    sprite_mask = agent_mask[y_min:y_max+1, x_min:x_max+1]
    sprite_pixels = cell[y_min:y_max+1, x_min:x_max+1]
    
    # 2. Create clean background
    clean_bg = img.copy()
    clean_bg[886:886+90, 421:421+90] = [50, 200, 50]
    
    # 3. Generate path and frames
    path = generate_path()
    frames_coords = []
    for _ in range(3):
        frames_coords.append(path[0])

    for i in range(len(path)-1):
        start_node = path[i]
        end_node = path[i+1]
        for step in range(4):
            frac = step / 4.0
            r = start_node[0] + (end_node[0] - start_node[0]) * frac
            c = start_node[1] + (end_node[1] - start_node[1]) * frac
            frames_coords.append((r, c))
        if i + 1 in [5, 14, 20]:
            frames_coords.append(end_node)

    frames_coords.append(path[-1])
    while len(frames_coords) < 102:
        frames_coords.append(path[-1])
        
    # 4. Generate video frames
    clean_bg_rgb = cv2.cvtColor(clean_bg, cv2.COLOR_BGR2RGB)
    sprite_pixels_rgb = cv2.cvtColor(sprite_pixels, cv2.COLOR_BGR2RGB)
    
    video_writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', format='FFMPEG', pixelformat='yuv420p', macro_block_size=None)
    
    for i, (r, c) in enumerate(frames_coords):
        if i == 0:
            frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            frame_rgb = clean_bg_rgb.copy()
            cy = int(round(93 * r + 93))
            cx = int(round(93 * c + 93))
            top_y = cy - 23
            top_x = cx - 23
            
            patch = frame_rgb[top_y:top_y+47, top_x:top_x+47]
            patch[sprite_mask] = sprite_pixels_rgb[sprite_mask]
            
        video_writer.append_data(frame_rgb)
        
    video_writer.close()

if __name__ == '__main__':
    main()
