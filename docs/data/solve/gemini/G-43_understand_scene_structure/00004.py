import cv2
import numpy as np
import imageio

def draw_partial_rect(img, pt1, pt2, color, thickness, progress):
    x, y = pt1
    w = pt2[0] - pt1[0]
    h = pt2[1] - pt1[1]
    
    total_len = 2 * (w + h)
    current_len = total_len * progress
    
    pts = [
        (x, y),
        (x + w, y),
        (x + w, y + h),
        (x, y + h),
        (x, y)
    ]
    
    drawn_len = 0
    for i in range(4):
        p1 = pts[i]
        p2 = pts[i+1]
        
        # length of this segment
        seg_len = w if i % 2 == 0 else h
        
        if drawn_len + seg_len <= current_len:
            # draw full segment
            cv2.line(img, p1, p2, color, thickness)
            drawn_len += seg_len
        else:
            # draw partial segment
            rem = current_len - drawn_len
            if rem > 0:
                ratio = rem / seg_len
                px = int(p1[0] + (p2[0] - p1[0]) * ratio)
                py = int(p1[1] + (p2[1] - p1[1]) * ratio)
                cv2.line(img, p1, (px, py), color, thickness)
            break

def main():
    first_frame = cv2.imread('/app/first_frame.png')
    # cv2 imread loads in BGR
    first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    
    x1, y1 = 64, 65
    w, h = 342, 326
    x2, y2 = x1 + w, y1 + h
    color = (0, 255, 0) # Green in RGB
    thickness = 6
    
    frames = []
    num_frames = 28
    
    for i in range(num_frames):
        img = first_frame_rgb.copy()
        progress = i / float(num_frames - 1)
        
        if progress > 0:
            draw_partial_rect(img, (x1, y1), (x2, y2), color, thickness, progress)
            
        frames.append(img)
        
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', pixelformat='yuv420p', macro_block_size=1)
    for frame in frames:
        writer.append_data(frame)
    writer.close()

if __name__ == '__main__':
    main()
