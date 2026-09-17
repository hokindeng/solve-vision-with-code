import cv2
import numpy as np
import imageio

def make_video():
    first_frame = cv2.imread('/app/first_frame.png')
    
    cell1 = first_frame[2:339, 342:681].copy()
    cell8_initial = first_frame[684:1021, 683:1022].copy()
    cell_blank = np.full_like(cell8_initial, 255)
    
    num_frames = 35
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', pixelformat='yuv420p')
    
    for i in range(num_frames):
        frame = first_frame.copy()
        
        if i == 0:
            current_cell = cell8_initial
        elif i <= 11:
            alpha = 1.0 - (i / 11.0)
            current_cell = cv2.addWeighted(cell8_initial, alpha, cell_blank, 1 - alpha, 0)
        elif i <= 16:
            current_cell = cell_blank
        elif i < num_frames - 1:
            alpha = (i - 17) / (num_frames - 1 - 17)
            current_cell = cv2.addWeighted(cell1, alpha, cell_blank, 1 - alpha, 0)
        else:
            current_cell = cell1
            
        frame[684:1021, 683:1022] = current_cell
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_rgb)
        
    writer.close()

if __name__ == '__main__':
    make_video()
