import cv2
import numpy as np
import imageio
import os

def solve():
    os.makedirs('/app/output', exist_ok=True)
    img = cv2.imread('/app/first_frame.png')
    
    # Extract D and create templates
    D_rect = (111, 602, 154, 145)
    x, y, w, h = D_rect
    D_crop = img[y:y+h, x:x+w]
    
    # Tolerant mask for green [71, 153, 30] (BGR)
    target_color = np.array([71, 153, 30])
    diff_green = np.abs(D_crop.astype(np.int32) - target_color)
    green_mask = (np.max(diff_green, axis=-1) < 15).astype(np.uint8)
    
    # Anti-aliased alpha by distance from white
    diff_white = 255 - D_crop
    max_diff = np.max(diff_white, axis=-1)
    # The max diff for the green color is 255 - 30 = 225
    alpha = np.clip(max_diff * (255.0 / 225.0), 0, 255).astype(np.uint8)
    
    extracted_D = np.zeros((h, w, 4), dtype=np.uint8)
    # Re-paint the interior with the exact green color to fix any artifacts
    # We mix original pixels with target_color based on how green they are?
    # No, let's just use the original image's pixels, they are fine.
    extracted_D[:,:,:3] = D_crop
    extracted_D[:,:,3] = alpha
    
    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    pad = 10
    template_shape = (h + 2*pad, w + 2*pad, 4)
    
    shifted_contours = [c + [pad, pad] for c in contours]
    
    outline_D = np.zeros(template_shape, dtype=np.uint8)
    cv2.drawContours(outline_D, shifted_contours, -1, (71, 153, 30, 255), 5, lineType=cv2.LINE_AA)
    
    # Base positions
    pos_E = (389, 592)
    pos_F = (677, 652)
    
    # ? patches
    patch1_rect = (450, 650, 50, 65) # x, y, w, h
    patch2_rect = (740, 650, 50, 65)
    
    def overlay(bg, fg, x, y, alpha=1.0):
        if alpha <= 0: return
        bg_h, bg_w = bg.shape[:2]
        fg_h, fg_w = fg.shape[:2]
        
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(bg_w, x + fg_w), min(bg_h, y + fg_h)
        
        if x1 >= x2 or y1 >= y2:
            return
            
        fg_x1 = x1 - x
        fg_y1 = y1 - y
        fg_x2 = fg_x1 + (x2 - x1)
        fg_y2 = fg_y1 + (y2 - y1)
        
        fg_crop = fg[fg_y1:fg_y2, fg_x1:fg_x2]
        bg_crop = bg[y1:y2, x1:x2]
        
        fg_alpha = (fg_crop[:, :, 3] / 255.0) * alpha
        fg_alpha = np.expand_dims(fg_alpha, axis=-1)
        
        res = bg_crop * (1 - fg_alpha) + fg_crop[:, :, :3] * fg_alpha
        bg[y1:y2, x1:x2] = res.astype(np.uint8)

    def fade_patch(bg, rect, alpha):
        if alpha <= 0: return
        px, py, pw, ph = rect
        bg_crop = bg[py:py+ph, px:px+pw]
        res = bg_crop * (1 - alpha) + 255.0 * alpha
        bg[py:py+ph, px:px+pw] = res.astype(np.uint8)
        
    frames = []
    
    for frame_idx in range(64):
        bg = img.copy()
        
        if frame_idx < 32:
            # Phase 1: 0 to 31
            t = frame_idx / 31.0
            
            fade_patch(bg, patch1_rect, t)
            
            curr_ex_x = int(round(111 + 288 * t))
            curr_ex_y = int(round(602))
            overlay(bg, extracted_D, curr_ex_x, curr_ex_y, 1.0 - t)
            
            curr_out_x = int(round(101 + 288 * t))
            curr_out_y = int(round(592))
            overlay(bg, outline_D, curr_out_x, curr_out_y, t)
            
        else:
            # Phase 2: 32 to 63
            t = (frame_idx - 32) / 31.0
            
            fade_patch(bg, patch1_rect, 1.0)
            overlay(bg, outline_D, pos_E[0], pos_E[1], 1.0)
            
            fade_patch(bg, patch2_rect, t)
            
            curr_out_x = int(round(pos_E[0] + 288 * t))
            curr_out_y = int(round(pos_E[1] + 60 * t))
            
            overlay(bg, outline_D, curr_out_x, curr_out_y, 1.0)
            
        frames.append(cv2.cvtColor(bg, cv2.COLOR_BGR2RGB))
        
    imageio.mimwrite('/app/output/video.mp4', frames, fps=16, codec='libx264', pixelformat='yuv420p')

if __name__ == '__main__':
    solve()
