import cv2
import numpy as np
import imageio
import os

def get_drawing_points(x_min, y_min, x_max, y_max, length):
    segments = []
    
    # top edge (left to right)
    l1 = x_max - x_min
    if length <= l1:
        segments.append( ((x_min, y_min), (x_min + int(length), y_min)) )
        return segments
    segments.append( ((x_min, y_min), (x_max, y_min)) )
    length -= l1
    
    # right edge (top to bottom)
    l2 = y_max - y_min
    if length <= l2:
        segments.append( ((x_max, y_min), (x_max, y_min + int(length))) )
        return segments
    segments.append( ((x_max, y_min), (x_max, y_max)) )
    length -= l2
    
    # bottom edge (right to left)
    l3 = x_max - x_min
    if length <= l3:
        segments.append( ((x_max, y_max), (x_max - int(length), y_max)) )
        return segments
    segments.append( ((x_max, y_max), (x_min, y_max)) )
    length -= l3
    
    # left edge (bottom to top)
    l4 = y_max - y_min
    if length <= l4:
        segments.append( ((x_min, y_max), (x_min, y_max - int(length))) )
        return segments
    segments.append( ((x_min, y_max), (x_min, y_min)) )
    
    return segments

def main():
    img = cv2.imread('/app/first_frame.png')
    
    # The maximum value point is the peak at X=552, Y=242.
    x_center = 552
    y_center = 242
    s = 20
    x_min, y_min = x_center - s, y_center - s
    x_max, y_max = x_center + s, y_center + s
    
    perimeter = (x_max - x_min) * 2 + (y_max - y_min) * 2
    
    frames = []
    num_frames = 48
    
    for t in range(num_frames):
        frame = img.copy()
        fraction = t / (num_frames - 1)
        length = perimeter * fraction
        
        segments = get_drawing_points(x_min, y_min, x_max, y_max, length)
        for pt1, pt2 in segments:
            cv2.line(frame, pt1, pt2, (0, 0, 255), 4, cv2.LINE_AA)
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        
    os.makedirs('/app/output', exist_ok=True)
    imageio.mimwrite('/app/output/video.mp4', frames, fps=16, codec='libx264', pixelformat='yuv420p')

if __name__ == '__main__':
    main()
