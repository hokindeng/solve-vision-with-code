import cv2
import numpy as np
import imageio
import os

def draw_partial_rect(img, x1, y1, x2, y2, color, thickness, progress):
    drawn_img = img.copy()
    
    if progress >= 1.0:
        cv2.rectangle(drawn_img, (x1, y1), (x2, y2), color, thickness)
        return drawn_img
        
    w = x2 - x1
    h = y2 - y1
    perimeter = 2 * w + 2 * h
    current_len = perimeter * progress
    
    segments = [
        ((x1, y1), (x2, y1), w),
        ((x2, y1), (x2, y2), h),
        ((x2, y2), (x1, y2), w),
        ((x1, y2), (x1, y1), h)
    ]
    
    for start_pt, end_pt, seg_len in segments:
        if current_len <= 0:
            break
        if current_len >= seg_len:
            cv2.line(drawn_img, start_pt, end_pt, color, thickness)
            current_len -= seg_len
        else:
            ratio = current_len / seg_len
            partial_pt = (
                int(start_pt[0] + (end_pt[0] - start_pt[0]) * ratio),
                int(start_pt[1] + (end_pt[1] - start_pt[1]) * ratio)
            )
            cv2.line(drawn_img, start_pt, partial_pt, color, thickness)
            current_len = 0
            
    return drawn_img

def main():
    os.makedirs('/app/output', exist_ok=True)
    img_bgr = cv2.imread('/app/first_frame.png')
    
    x1, y1 = 60, 60
    x2, y2 = 305, 570
    
    color_bgr = (0, 255, 0)
    thickness = 8
    
    num_frames = 28
    fps = 16
    
    writer = imageio.get_writer('/app/output/video.mp4', fps=fps, codec='libx264', pixelformat='yuv420p')
    
    for i in range(num_frames):
        progress = i / (num_frames - 1)
        frame_bgr = draw_partial_rect(img_bgr, x1, y1, x2, y2, color_bgr, thickness, progress)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_rgb)
        
    writer.close()

if __name__ == "__main__":
    main()
