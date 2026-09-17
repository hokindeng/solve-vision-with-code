import os
import cv2
import numpy as np
import imageio

def solve():
    img_path = '/app/first_frame.png'
    out_dir = '/app/output'
    out_path = os.path.join(out_dir, 'video.mp4')
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image at {img_path}")

    # Threshold to find the black borders
    mask = cv2.inRange(img, (0, 0, 0), (10, 10, 10))
    
    # Find external contours of the borders
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours systematically (top to bottom, left to right)
    contours = list(contours)
    contours.sort(key=lambda c: (cv2.boundingRect(c)[1], cv2.boundingRect(c)[0]))
    
    N = len(contours)
    
    fps = 16
    total_frames = 30
    
    writer = imageio.get_writer(out_path, fps=fps, codec='libx264', pixelformat='yuv420p', macro_block_size=None)
    
    if N == 0:
        start_frames = 0
        frames_per_object = 0
        end_frames = total_frames
    else:
        start_frames = max(2, int(total_frames * 0.1))
        min_end_frames = max(5, int(total_frames * 0.25))
        
        count_frames = total_frames - start_frames - min_end_frames
        
        frames_per_object = max(1, count_frames // N)
        actual_count_frames = frames_per_object * N
        
        end_frames = total_frames - start_frames - actual_count_frames
        if end_frames < min_end_frames:
            end_frames = min_end_frames
            
    # Write start frames
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for _ in range(start_frames):
        writer.append_data(img_rgb)
        
    # Write counting frames
    for cnt in contours:
        frame = img.copy()
        # Highlight border in Yellow (BGR: 0, 255, 255)
        cv2.drawContours(frame, [cnt], 0, (0, 255, 255), 6)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        for _ in range(frames_per_object):
            writer.append_data(frame_rgb)
            
    # Write end frames with "Count: N"
    final_frame = img.copy()
    text = f"Count: {N}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 3
    font_thickness = 6
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    
    text_x = (final_frame.shape[1] - text_size[0]) // 2
    text_y = (final_frame.shape[0] + text_size[1]) // 2
    
    cv2.putText(final_frame, text, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness)
    final_frame_rgb = cv2.cvtColor(final_frame, cv2.COLOR_BGR2RGB)
    
    for _ in range(end_frames):
        writer.append_data(final_frame_rgb)
        
    writer.close()

if __name__ == '__main__':
    solve()
