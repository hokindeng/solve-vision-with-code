import cv2
import numpy as np
import imageio
import os

def main():
    first_frame = cv2.imread('/app/first_frame.png')
    
    # BGR colors
    color_333 = np.array([229, 80, 80], dtype=np.float32)
    color_295 = np.array([153, 103, 53], dtype=np.float32)
    
    # Extract octagon template from D (x: 90 to 271, y: 592 to 773)
    D_img = first_frame[592:592+181, 90:90+181]
    
    os.makedirs('/app/output', exist_ok=True)
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', macro_block_size=None, pixelformat='yuv420p')
    
    def draw_octagon(frame, center_x, center_y, size, color):
        resized_D = cv2.resize(D_img, (size, size), interpolation=cv2.INTER_NEAREST)
        border = (resized_D == [0, 0, 0]).all(axis=2)
        interior = (resized_D == [229, 80, 80]).all(axis=2)
        
        x0 = int(round(center_x - size / 2))
        y0 = int(round(center_y - size / 2))
        
        for c in range(3):
            frame_slice = frame[y0:y0+size, x0:x0+size, c]
            frame_slice[interior] = color[c]
            frame_slice[border] = 0

    for f in range(60):
        frame = first_frame.copy()
        
        if f == 0:
            # First frame must remain unchanged as per instructions.
            pass
        elif f < 30:
            # Step 1: E appears and changes color
            # Erase ? at E
            frame[592:773, 428:609] = 255
            
            alpha = f / 29.0
            curr_color = color_333 * (1 - alpha) + color_295 * alpha
            curr_color = np.round(curr_color).astype(int)
            
            draw_octagon(frame, 518.5, 682.5, 181, curr_color)
            
        else:
            # Step 1 finished, E is static
            frame[592:773, 428:609] = 255
            draw_octagon(frame, 518.5, 682.5, 181, color_295)
            
            # Step 2: F appears and changes size
            # Erase ? at F
            frame[592:773, 766:947] = 255
            
            alpha = (f - 30) / 29.0
            curr_size = int(round(181 - alpha * (181 - 141)))
            
            draw_octagon(frame, 856.5, 682.5, curr_size, color_295)
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_rgb)
        
    writer.close()

if __name__ == '__main__':
    main()
