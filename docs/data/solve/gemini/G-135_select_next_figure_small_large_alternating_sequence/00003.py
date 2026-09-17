import imageio
import cv2
import numpy as np

def make_video():
    first_frame = cv2.imread('/app/first_frame.png')
    first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    
    height, width, _ = first_frame_rgb.shape
    
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', macro_block_size=None, pixelformat='yuv420p')
    
    center = (626, 854)
    radius = 95
    thickness = 8
    color = (255, 0, 0) # Red in RGB
    
    num_frames = 60
    
    for i in range(num_frames):
        frame = first_frame_rgb.copy()
        
        angle = int(360 * (i + 1) / num_frames)
        
        if angle > 0:
            # OpenCV ellipse works with BGR/RGB as long as we pass the right color tuple
            # The image here is in RGB, so color is (255, 0, 0)
            cv2.ellipse(frame, center, (radius, radius), 0, -90, -90 + angle, color, thickness, cv2.LINE_AA)
            
        writer.append_data(frame)
        
    writer.close()

if __name__ == '__main__':
    make_video()
