import cv2
import numpy as np
import imageio
import os

def get_full_brick_and_bg(img):
    callout = img[332:498, 72:236]
    pure_red = cv2.inRange(callout, np.array([0,0,255]), np.array([0,0,255]))
    kernel = np.ones((3,3), np.uint8)
    pure_red_dil = cv2.dilate(pure_red, kernel, iterations=1)
    
    mask = np.zeros(callout.shape[:2], np.uint8)
    mask[:] = cv2.GC_PR_BGD
    
    white_mask = cv2.inRange(callout, np.array([245,245,245]), np.array([255,255,255]))
    mask[white_mask > 0] = cv2.GC_BGD
    mask[pure_red_dil > 0] = cv2.GC_BGD
    
    hsv = cv2.cvtColor(callout, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    mask2_red = cv2.inRange(hsv, lower_red2, upper_red2)
    # Exclude pure red arrow from red mask!
    red_mask = (mask1 | mask2_red) & (pure_red == 0)
    mask[red_mask > 0] = cv2.GC_FGD
    
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    cv2.grabCut(callout, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
    brick_alpha = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
    
    clean_bg = img.copy()
    callout_bg = clean_bg[332:498, 72:236]
    callout_bg[brick_alpha > 0] = [255, 255, 255]
    
    callout_inpainted = cv2.inpaint(callout, pure_red_dil, 3, cv2.INPAINT_TELEA)
    
    mask_full = np.zeros(callout.shape[:2], np.uint8)
    mask_full[:] = cv2.GC_PR_BGD
    white_mask_inp = cv2.inRange(callout_inpainted, np.array([245,245,245]), np.array([255,255,255]))
    mask_full[white_mask_inp > 0] = cv2.GC_BGD
    
    # Re-calculate red_mask on inpainted image without excluding pure_red (since it's gone)
    hsv_inp = cv2.cvtColor(callout_inpainted, cv2.COLOR_BGR2HSV)
    mask1_inp = cv2.inRange(hsv_inp, lower_red1, upper_red1)
    mask2_inp = cv2.inRange(hsv_inp, lower_red2, upper_red2)
    red_mask_inp = mask1_inp | mask2_inp
    mask_full[red_mask_inp > 0] = cv2.GC_FGD
    
    cv2.grabCut(callout_inpainted, mask_full, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
    full_alpha = np.where((mask_full==2)|(mask_full==0), 0, 1).astype('uint8')
    
    full_brick = cv2.cvtColor(callout_inpainted, cv2.COLOR_BGR2BGRA)
    full_brick[:, :, 3] = full_alpha * 255
    
    return clean_bg, full_brick

def main():
    os.makedirs('/app/output', exist_ok=True)
    img = cv2.imread('/app/first_frame.png')
    clean_bg, full_brick = get_full_brick_and_bg(img)
    
    frames = []
    
    x0, y0 = 72, 332
    dx_total, dy_total = 432, 475
    
    num_frames = 46
    move_frames = 40
    
    h, w = full_brick.shape[:2]
    alpha = full_brick[:, :, 3] / 255.0
    
    for i in range(num_frames):
        if i == 0:
            frame = img.copy()
        else:
            frame = clean_bg.copy()
            
            t = min(i / move_frames, 1.0)
            ease_t = t * t * (3 - 2 * t)
            
            cx = int(x0 + dx_total * ease_t)
            cy = int(y0 + dy_total * ease_t)
            
            # fast alpha blending
            for c in range(3):
                frame[cy:cy+h, cx:cx+w, c] = (frame[cy:cy+h, cx:cx+w, c] * (1 - alpha) + full_brick[:, :, c] * alpha).astype(np.uint8)
                
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', pixelformat='yuv420p', macro_block_size=None)
    for f in frames:
        writer.append_data(f)
    writer.close()
    
    print("Video generated successfully.")

if __name__ == '__main__':
    main()
