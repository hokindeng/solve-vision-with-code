import cv2
import numpy as np
import imageio
import os

def solve():
    img = cv2.imread('/app/first_frame.png')
    
    # Find non-white pixels
    non_white = (img[:,:,0] < 250) | (img[:,:,1] < 250) | (img[:,:,2] < 250)
    thresh = np.uint8(non_white * 255)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the contour that contains red pixels
    target_rect = None
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        roi = img[y:y+h, x:x+w]
        # Check if there are red pixels in this roi
        red_mask = (roi[:,:,2] > 200) & (roi[:,:,1] < 50) & (roi[:,:,0] < 50)
        if np.any(red_mask):
            target_rect = (x, y, w, h)
            break
            
    if target_rect is None:
        print("Could not find target symbol.")
        return
        
    tx, ty, tw, th = target_rect
    
    os.makedirs('/app/output', exist_ok=True)
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', pixelformat='yuv420p', macro_block_size=None)
    
    n_frames = 46
    
    # To be safe and capture any anti-aliasing, we can pad slightly if space allows
    pad = 2
    y1 = max(0, ty - pad)
    y2 = min(img.shape[0], ty + th + pad)
    x1 = max(0, tx - pad)
    x2 = min(img.shape[1], tx + tw + pad)
    
    for i in range(n_frames):
        frame = img.copy()
        
        alpha = i / (n_frames - 1)
        
        region = frame[y1:y2, x1:x2].astype(np.float32)
        white = np.full_like(region, 255.0)
        
        blended = region * (1 - alpha) + white * alpha
        frame[y1:y2, x1:x2] = blended.astype(np.uint8)
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_rgb)
        
    writer.close()

if __name__ == '__main__':
    solve()
