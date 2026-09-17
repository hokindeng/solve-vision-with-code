import cv2
import numpy as np
import imageio

def draw_partial_box(img, x1, y1, x2, y2, color, thickness, progress):
    """
    progress: float from 0.0 to 1.0
    Draws a rectangle starting from top-left, going clockwise.
    """
    w = x2 - x1
    h = y2 - y1
    total_len = 2 * w + 2 * h
    target_len = int(progress * total_len)
    
    drawn_img = img.copy()
    
    # We will draw lines. But since we need thickness, we should just draw the segments that fit within target_len.
    # To make the corners look good, we can just create an empty mask, draw the lines on the mask, and then alpha composite or just copy over.
    # Since it's a simple solid color replacement without alpha (or maybe we should just draw it directly).
    # Actually, drawing it exactly by filling the rectangles is better.
    
    current_len = 0
    
    # Top edge
    if target_len > current_len:
        draw_len = min(target_len - current_len, w)
        cv2.rectangle(drawn_img, (x1, y1), (x1 + draw_len, y1 + thickness - 1), color, -1)
        current_len += w

    # Right edge
    if target_len > current_len:
        draw_len = min(target_len - current_len, h)
        cv2.rectangle(drawn_img, (x2 - thickness + 1, y1), (x2, y1 + draw_len), color, -1)
        current_len += h
        
    # Bottom edge (drawing from right to left)
    if target_len > current_len:
        draw_len = min(target_len - current_len, w)
        cv2.rectangle(drawn_img, (x2 - draw_len, y2 - thickness + 1), (x2, y2), color, -1)
        current_len += w
        
    # Left edge (drawing from bottom to top)
    if target_len > current_len:
        draw_len = min(target_len - current_len, h)
        cv2.rectangle(drawn_img, (x1, y2 - draw_len), (x1 + thickness - 1, y2), color, -1)
        current_len += h

    return drawn_img

def main():
    img = cv2.imread('/app/first_frame.png')
    
    x1, y1 = 65, 335
    x2, y2 = 509, 628
    thickness = 6
    color = (0, 255, 0) # Green in BGR, but imageio expects RGB. We will convert at the end.
    
    frames = []
    num_frames = 28
    
    for i in range(num_frames):
        progress = i / (num_frames - 1)
        frame_bgr = draw_partial_box(img, x1, y1, x2, y2, color, thickness, progress)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        
    import os
    os.makedirs('/app/output', exist_ok=True)
    
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, macro_block_size=None, format='FFMPEG', codec='libx264', pixelformat='yuv420p')
    for frame in frames:
        writer.append_data(frame)
    writer.close()

if __name__ == '__main__':
    main()
