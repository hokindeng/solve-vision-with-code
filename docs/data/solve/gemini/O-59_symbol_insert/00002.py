import cv2
import numpy as np
import imageio

def ease_in_out(t):
    return t * t * (3 - 2 * t)

def main():
    img = cv2.imread('/app/first_frame.png')
    
    # Extract components
    empty_slot = img[464:464+97, 831:831+97].copy()
    ref = img[18:18+120, 887:887+120]
    diamond = ref[20:98, 20:98].copy()
    
    # Create new symbol
    new_symbol = empty_slot.copy()
    new_symbol[9:87, 9:87] = diamond
    
    # Tiles to move
    tile6 = img[464:464+97, 621:621+97].copy()
    tile7 = img[464:464+97, 726:726+97].copy()
    
    # Base frame (with empty slots at 6 and 7)
    base_frame = img.copy()
    base_frame[464:464+97, 621:621+97] = empty_slot
    base_frame[464:464+97, 726:726+97] = empty_slot
    
    frames = []
    
    for f in range(32):
        if f <= 11:
            t1 = f / 11.0
            t2 = 0.0
            t3 = 0.0
        elif f <= 21:
            t1 = 1.0
            t2 = (f - 11) / 10.0
            t3 = 0.0
        else:
            t1 = 1.0
            t2 = 1.0
            t3 = (f - 21) / 10.0
            
        t1_e = ease_in_out(t1)
        t3_e = ease_in_out(t3)
        
        frame = base_frame.copy()
        
        # Draw moving tiles
        dx = int(round(105 * t1_e))
        
        x6 = 621 + dx
        x7 = 726 + dx
        
        frame[464:464+97, x6:x6+97] = tile6
        frame[464:464+97, x7:x7+97] = tile7
        
        # Draw new symbol
        if t2 > 0:
            y_start = 354
            y_end = 464
            y_curr = int(round(y_start + (y_end - y_start) * t3_e))
            x_curr = 621
            
            patch = frame[y_curr:y_curr+97, x_curr:x_curr+97]
            blended = cv2.addWeighted(new_symbol, t2, patch, 1 - t2, 0)
            frame[y_curr:y_curr+97, x_curr:x_curr+97] = blended
            
        # Convert BGR to RGB for imageio
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        
    # Write video
    imageio.mimwrite('/app/output/video.mp4', frames, fps=16, codec='libx264', pixelformat='yuv420p')

if __name__ == '__main__':
    main()
