import cv2
import numpy as np
import imageio
import os

def solve():
    img = cv2.imread('/app/first_frame.png')
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 80, 175, 76 in BGR is 76, 175, 80 in RGB
    cell = img_rgb[2:255, 2:255]
    green = np.all(cell == [76, 175, 80], axis=2)
    mask = ~green
    
    sprite_img = cell[82:211, 62:191].copy()
    sprite_mask = mask[82:211, 62:191].copy()
    
    bg = img_rgb.copy()
    bg[2:255, 2:255] = [76, 175, 80]
    
    def draw_pacman(canvas, cx, cy, k):
        x0, y0 = int(round(cx)) - 64, int(round(cy)) - 64
        x1, y1 = x0 + 129, y0 + 129
        
        rot_img = np.rot90(sprite_img, k)
        rot_mask = np.rot90(sprite_mask, k)
        
        region = canvas[y0:y1, x0:x1]
        np.copyto(region, rot_img, where=rot_mask[..., None])

    path = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (3, 1), (3, 2), (2, 2), (2, 1),
        (1, 1), (0, 1), (0, 2), (0, 3),
        (1, 3), (2, 3), (3, 3)
    ]
    
    frames_list = []
    
    for step in range(len(path) - 1):
        x0, y0 = path[step]
        x1, y1 = path[step+1]
        
        cx0 = x0 * 256 + 128
        cy0 = y0 * 256 + 148
        cx1 = x1 * 256 + 128
        cy1 = y1 * 256 + 148
        
        # Determine orientation
        if x1 > x0: k = 0
        elif y1 > y0: k = -1
        elif x1 < x0: k = 2
        elif y1 < y0: k = 1
        
        # 6 frames per move. Frame 0 of this move, up to Frame 5.
        # Frame 6 will be generated as Frame 0 of the next move.
        for f in range(6):
            t = f / 6.0
            cx = cx0 + (cx1 - cx0) * t
            cy = cy0 + (cy1 - cy0) * t
            
            canvas = bg.copy()
            draw_pacman(canvas, cx, cy, k)
            frames_list.append(canvas)
            
    # Add the final position frames
    final_x, final_y = path[-1]
    final_cx = final_x * 256 + 128
    final_cy = final_y * 256 + 148
    # k is the orientation of the last move
    k = 0 # from (2,3) to (3,3) is Right, so k=0
    
    # We generated 14 * 6 = 84 frames.
    # We need 91 frames, so add 7 more frames (84 to 90) at the final position.
    for f in range(7):
        canvas = bg.copy()
        draw_pacman(canvas, final_cx, final_cy, k)
        frames_list.append(canvas)
        
    print(f"Generated {len(frames_list)} frames.")
    
    os.makedirs('/app/output', exist_ok=True)
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', pixelformat='yuv420p', macro_block_size=None)
    for frame in frames_list:
        writer.append_data(frame)
    writer.close()
    print("Video saved to /app/output/video.mp4")

if __name__ == '__main__':
    solve()
