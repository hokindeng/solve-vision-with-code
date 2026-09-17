import cv2
import numpy as np
import imageio
import os

def solve():
    img_first = cv2.imread('/app/first_frame.png')
    
    out_path = '/app/output/video.mp4'
    writer = imageio.get_writer(out_path, fps=16, macro_block_size=None, codec='libx264', pixelformat='yuv420p')
    
    max_r_tri = 80 / (1 + np.sqrt(2))
    
    for frame in range(60):
        if frame == 0:
            # First frame is exactly first_frame.png
            img_rgb = cv2.cvtColor(img_first, cv2.COLOR_BGR2RGB)
            writer.append_data(img_rgb)
            continue
            
        img = img_first.copy()
        # Erase the ?
        img[650:850, 750:950] = (255, 255, 255)
        
        # Calculate p
        p = (frame - 1) / 58.0
        p = min(max(p, 0.0), 1.0)
        
        # Ease-in-out could be nice, but linear is fine too. Let's use linear.
        
        r_out = 4 * p
        shift_x = 4 * p
        r_in_rect = 40 * (1 - p) + 4 * p
        r_in_tri = max_r_tri * (1 - p) + 4 * p
        
        def get_rect(r, shift):
            left = 780 - r + shift
            right = 860 + r + shift
            top = 728 - r
            bottom = 808 + r
            return np.array([
                [left, top],
                [right, top],
                [right, bottom],
                [left, bottom]
            ], dtype=np.float32)

        def get_tri(r, shift):
            x_left = 860 - r + shift
            x_right = 940 + r * np.sqrt(2) + shift
            y_top = 688 - r * (1 + np.sqrt(2))
            y_bot = 848 + r * (1 + np.sqrt(2))
            return np.array([
                [x_left, y_top],
                [x_right, 768],
                [x_left, y_bot]
            ], dtype=np.float32)

        rect_out = get_rect(r_out, shift_x)
        tri_out = get_tri(r_out, shift_x)
        
        rect_in = get_rect(-r_in_rect, shift_x)
        tri_in = get_tri(-r_in_tri, shift_x)
        
        # We need to round to int for cv2.fillPoly
        # but to get better anti-aliasing, we can use fractional coordinates 
        # using the shift parameter in fillPoly.
        # fractional bits = 4 (subpixel accuracy)
        shift_bits = 4
        scale = 2 ** shift_bits
        
        rect_out = np.round(rect_out * scale).astype(np.int32)
        tri_out = np.round(tri_out * scale).astype(np.int32)
        rect_in = np.round(rect_in * scale).astype(np.int32)
        tri_in = np.round(tri_in * scale).astype(np.int32)
        
        color = (191, 108, 66) # BGR
        white = (255, 255, 255)
        
        # Draw outer shapes
        cv2.fillPoly(img, [rect_out], color, lineType=cv2.LINE_AA, shift=shift_bits)
        cv2.fillPoly(img, [tri_out], color, lineType=cv2.LINE_AA, shift=shift_bits)
        
        # Draw inner shapes in white to create the hole
        cv2.fillPoly(img, [rect_in], white, lineType=cv2.LINE_AA, shift=shift_bits)
        cv2.fillPoly(img, [tri_in], white, lineType=cv2.LINE_AA, shift=shift_bits)
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        writer.append_data(img_rgb)
        
    writer.close()

if __name__ == '__main__':
    solve()
