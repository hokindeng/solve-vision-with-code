import cv2
import numpy as np
import imageio

def create_video():
    img = cv2.imread('/app/first_frame.png')
    
    # 1. Identify blocks
    blocks = []
    for y in range(16):
        for x in range(16):
            if np.array_equal(img[y*64+32, x*64+32], [0, 255, 0]):
                blocks.append((x, y))
                
    # 2. Extract background
    bg = img.copy()
    for (x, y) in blocks:
        bg[y*64+1:(y+1)*64, x*64+1:(x+1)*64] = [255, 255, 255]
        
    # 3. Extract block sprite (54x54)
    bx, by = blocks[0]
    sprite = img[by*64+5:by*64+59, bx*64+5:bx*64+59].copy()
    
    # 4. Generate frames
    num_frames = 35
    fps = 16
    
    writer = imageio.get_writer('/app/output/video.mp4', fps=fps, codec='libx264', pixelformat='yuv420p', macro_block_size=None, quality=10)
    
    for i in range(num_frames):
        frame = bg.copy()
        
        # Calculate offset
        dy = int(round(-64.0 * i / (num_frames - 1)))
        
        for (x, y) in blocks:
            y_pos = y*64 + 5 + dy
            x_pos = x*64 + 5
            
            # Since blocks might move up, handle drawing
            frame[y_pos:y_pos+54, x_pos:x_pos+54] = sprite
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_rgb)
        
    writer.close()

if __name__ == '__main__':
    create_video()
