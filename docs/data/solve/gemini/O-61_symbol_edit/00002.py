import cv2
import numpy as np
import imageio
import math

def ease(t):
    if t <= 0: return 0.0
    if t >= 1: return 1.0
    return (1 - math.cos(math.pi * t)) / 2.0

def paste_symbol(frame, mask_img, symbol_img, x, y, alpha=1.0):
    if alpha <= 0: return
    x = int(round(x))
    y = int(round(y))
    patch = frame[y:y+97, x:x+97].astype(np.float32)
    alpha_mask = (mask_img * alpha)[..., None]
    blended = patch * (1 - alpha_mask) + symbol_img.astype(np.float32) * alpha_mask
    frame[y:y+97, x:x+97] = blended.astype(np.uint8)

def main():
    img = cv2.imread('/app/first_frame.png')
    
    # We found earlier that slot 6 (empty) starts at X=569
    empty = img[464:464+97, 569:569+97].copy()
    slots_x = [44, 149, 254, 359, 464, 569, 674, 779, 884]
    
    def extract_symbol(idx):
        x = slots_x[idx]
        roi = img[464:464+97, x:x+97].copy()
        diff = cv2.absdiff(roi, empty)
        mask = np.any(diff > 10, axis=-1).astype(np.float32)
        return mask, roi

    mask3, sym3 = extract_symbol(2) # S3
    mask4, sym4 = extract_symbol(3) # S4
    mask5, sym5 = extract_symbol(4) # S5 (Hollow Diamond)
    
    base_img = img.copy()
    # Erase S3, S4, S5
    for i in [2, 3, 4]:
        x = slots_x[i]
        base_img[464:464+97, x:x+97] = empty
        
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', pixelformat='yuv420p')
    
    num_frames = 76
    
    for f in range(num_frames):
        frame = base_img.copy()
        
        # Movements: start at 10, end at 40
        t_move = (f - 10) / 30.0
        p_move = ease(t_move)
        
        # Fade-ins: start at 40, end at 65
        t_fade = (f - 40) / 25.0
        p_fade = ease(t_fade)
        
        # Positions
        x3_curr = slots_x[2] + p_move * (slots_x[3] - slots_x[2])
        x4_curr = slots_x[3] + p_move * (slots_x[4] - slots_x[3])
        x5_curr = slots_x[4] + p_move * (slots_x[6] - slots_x[4])
        
        paste_symbol(frame, mask3, sym3, x3_curr, 464)
        paste_symbol(frame, mask4, sym4, x4_curr, 464)
        paste_symbol(frame, mask5, sym5, x5_curr, 464)
        
        paste_symbol(frame, mask5, sym5, slots_x[2], 464, alpha=p_fade) # Pos 3
        paste_symbol(frame, mask5, sym5, slots_x[5], 464, alpha=p_fade) # Pos 6
        paste_symbol(frame, mask5, sym5, slots_x[7], 464, alpha=p_fade) # Pos 8
        
        # RGB for imageio
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_rgb)
        
    writer.close()
    print("Video generated!")

if __name__ == '__main__':
    main()

