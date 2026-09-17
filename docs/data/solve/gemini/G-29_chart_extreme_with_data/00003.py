import cv2
import numpy as np
import imageio
import os

def draw_partial_rect(img, top_left, bottom_right, length, color, thickness=3):
    x1, y1 = top_left
    x2, y2 = bottom_right
    w = x2 - x1
    h = y2 - y1
    
    # Create a full rect drawing
    rect_img = np.zeros_like(img)
    cv2.rectangle(rect_img, (x1, y1), (x2, y2), color, thickness)
    
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    th = thickness // 2 + 2 # padding for the mask to cover line thickness
    
    current_len = 0
    # Top edge
    l = w
    if current_len < length:
        draw_l = min(l, length - current_len)
        cv2.rectangle(mask, (x1 - th, y1 - th), (int(x1 + draw_l) + th, y1 + th), 255, -1)
        current_len += l
        
    # Right edge
    l = h
    if current_len < length:
        draw_l = min(l, length - current_len)
        cv2.rectangle(mask, (x2 - th, y1 - th), (x2 + th, int(y1 + draw_l) + th), 255, -1)
        current_len += l
        
    # Bottom edge (draws from right to left)
    l = w
    if current_len < length:
        draw_l = min(l, length - current_len)
        cv2.rectangle(mask, (int(x2 - draw_l) - th, y2 - th), (x2 + th, y2 + th), 255, -1)
        current_len += l
        
    # Left edge (draws from bottom to top)
    l = h
    if current_len < length:
        draw_l = min(l, length - current_len)
        cv2.rectangle(mask, (x1 - th, int(y2 - draw_l) - th), (x1 + th, y2 + th), 255, -1)
        current_len += l
        
    # Copy the masked drawn rect onto the original image
    # Where mask > 0 and rect_img > 0
    drawn_pixels = (mask > 0) & (rect_img[:, :, 2] == 255) # since color is red (0,0,255) in BGR
    img[drawn_pixels] = color

def make_video():
    first_frame = cv2.imread('/app/first_frame.png')
    
    os.makedirs('/app/output', exist_ok=True)
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', pixelformat='yuv420p', macro_block_size=None)
    
    # center of the minimum value point
    center = (773, 800)
    w, h = 40, 40
    
    top_left = (center[0] - w//2, center[1] - h//2)
    bottom_right = (center[0] + w//2, center[1] + h//2)
    
    total_len = w * 2 + h * 2
    num_frames = 48
    
    for i in range(num_frames):
        frame = first_frame.copy()
        
        progress = i / (num_frames - 1)
        length_to_draw = progress * total_len
        
        if length_to_draw > 0:
            draw_partial_rect(frame, top_left, bottom_right, length_to_draw, (0, 0, 255), thickness=3)
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_rgb)
        
    writer.close()

if __name__ == '__main__':
    make_video()
