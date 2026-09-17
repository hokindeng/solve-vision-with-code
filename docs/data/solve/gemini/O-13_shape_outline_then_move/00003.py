import cv2
import numpy as np
import imageio

def make_video():
    first_frame = cv2.imread('/app/first_frame.png')
    
    # Yellow color
    yellow = np.array([45, 229, 229], dtype=np.uint8)
    white = np.array([255, 255, 255], dtype=np.uint8)
    
    # Bounding box of D
    x0, y0 = 108, 602
    w, h = 161, 161
    
    mask = cv2.inRange(first_frame, np.array([40, 220, 220]), np.array([50, 240, 240]))
    plus_mask = mask[y0:y0+h, x0:x0+w]
    
    k11 = np.ones((11, 11), np.uint8)
    inner_mask = cv2.erode(plus_mask, k11, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    
    bg = first_frame.copy()
    D_mask_full = np.zeros_like(mask)
    D_mask_full[y0:y0+h, x0:x0+w] = plus_mask
    bg[D_mask_full > 0] = 255
    
    frames = []
    
    # Phase 1: 0 to 31
    for f in range(32):
        frame = bg.copy()
        alpha = f / 31.0  # 0 to 1
        
        patch = np.full((h, w, 3), 255, dtype=np.uint8)
        
        outline = plus_mask - inner_mask
        patch[outline > 0] = yellow
        
        blended = yellow * (1 - alpha) + white * alpha
        patch[inner_mask > 0] = blended.astype(np.uint8)
        
        frame[y0:y0+h, x0:x0+w][plus_mask > 0] = patch[plus_mask > 0]
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        
    # Phase 2: 32 to 63
    for f in range(32, 64):
        frame = bg.copy()
        t = (f - 32) / 31.0 # 0 to 1
        current_y = int(np.round(y0 + t * 100))
        
        patch = np.full((h, w, 3), 255, dtype=np.uint8)
        outline = plus_mask - inner_mask
        patch[outline > 0] = yellow
        
        frame[current_y:current_y+h, x0:x0+w][outline > 0] = patch[outline > 0]
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        
    imageio.mimwrite('/app/output/video.mp4', frames, fps=16, codec='libx264', pixelformat='yuv420p')

if __name__ == "__main__":
    make_video()
