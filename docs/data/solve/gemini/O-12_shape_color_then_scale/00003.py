import cv2
import numpy as np
import imageio

def solve():
    # Read first frame
    first_frame = cv2.imread('/app/first_frame.png')
    
    # Extract D's alpha mask
    # D's bounding box: Rect 110, 612, 141, 141
    D_crop = first_frame[612:612+141, 110:110+141].astype(np.float32)
    C0 = np.array([153.0, 30.0, 132.0]) # color_394 in BGR
    bg = np.array([255.0, 255.0, 255.0])
    
    diff = bg - C0
    diff[diff == 0] = 1 # avoid div by zero, though it's not zero here
    
    alpha = (bg - D_crop) / diff
    alpha = np.clip(alpha, 0, 1)
    # mean across channels
    alpha = np.mean(alpha, axis=-1)
    
    # Prepare backgrounds
    # Erase QM1 (at E)
    bg_phase1 = first_frame.copy()
    bg_phase1[650:710, 500:540] = 255
    
    # Erase QM2 (at F) as well
    bg_phase2 = bg_phase1.copy()
    bg_phase2[650:710, 835:875] = 255
    
    c394 = np.array([153.0, 30.0, 132.0])
    c130 = np.array([30.0, 153.0, 132.0])
    
    frames = []
    
    for i in range(60):
        if i == 0:
            frame = first_frame.copy().astype(np.float32)
        elif i < 30:
            t = (i - 1) / 28.0
            t = np.clip(t, 0, 1)
            frame = bg_phase1.copy().astype(np.float32)
            
            # Draw E
            color = c394 * (1 - t) + c130 * t
            scale = 1.0
            cx, cy = 518.5, 682.5
            
            M = np.array([
                [scale, 0, cx - 70.5 * scale],
                [0, scale, cy - 70.5 * scale]
            ], dtype=np.float32)
            
            full_alpha = cv2.warpAffine(alpha, M, (1024, 1024), flags=cv2.INTER_AREA)
            color_img = np.zeros((1024, 1024, 3), dtype=np.float32)
            color_img[:] = color
            
            frame = color_img * full_alpha[..., None] + frame * (1 - full_alpha[..., None])
            
        else:
            t = (i - 30) / 29.0
            t = np.clip(t, 0, 1)
            frame = bg_phase2.copy().astype(np.float32)
            
            # Draw E (static at c130)
            color = c130
            scale = 1.0
            cx, cy = 518.5, 682.5
            M = np.array([
                [scale, 0, cx - 70.5 * scale],
                [0, scale, cy - 70.5 * scale]
            ], dtype=np.float32)
            full_alpha_E = cv2.warpAffine(alpha, M, (1024, 1024), flags=cv2.INTER_AREA)
            color_img_E = np.zeros((1024, 1024, 3), dtype=np.float32)
            color_img_E[:] = color
            frame = color_img_E * full_alpha_E[..., None] + frame * (1 - full_alpha_E[..., None])
            
            # Draw F (scaling down)
            color_F = c130
            scale_F = 1.0 * (1 - t) + (101.0 / 141.0) * t
            cx_F, cy_F = 854.5, 682.5
            M_F = np.array([
                [scale_F, 0, cx_F - 70.5 * scale_F],
                [0, scale_F, cy_F - 70.5 * scale_F]
            ], dtype=np.float32)
            full_alpha_F = cv2.warpAffine(alpha, M_F, (1024, 1024), flags=cv2.INTER_AREA)
            color_img_F = np.zeros((1024, 1024, 3), dtype=np.float32)
            color_img_F[:] = color_F
            
            frame = color_img_F * full_alpha_F[..., None] + frame * (1 - full_alpha_F[..., None])
            
        frame_uint8 = frame.astype(np.uint8)
        frame_rgb = cv2.cvtColor(frame_uint8, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        
    imageio.mimwrite('/app/output/video.mp4', frames, fps=16, macro_block_size=1)

if __name__ == '__main__':
    # create output folder if it doesn't exist
    import os
    os.makedirs('/app/output', exist_ok=True)
    solve()
