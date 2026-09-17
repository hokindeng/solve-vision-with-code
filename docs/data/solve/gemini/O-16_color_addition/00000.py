import cv2
import numpy as np
import imageio

def make_video():
    img = cv2.imread('/app/first_frame.png')
    
    # We found the balls are bounded at these coordinates:
    # Ball 1 center: (401, 294), box: [174:415, 281:522]
    # Ball 2 center: (444, 724), box: [604:845, 324:565]
    s1 = img[174:415, 281:522].copy()
    m1 = np.any(s1 != [255, 255, 255], axis=-1)
    
    s2 = img[604:845, 324:565].copy()
    m2 = np.any(s2 != [255, 255, 255], axis=-1)
    
    frames = []
    
    for t in range(80):
        # Interpolate centers
        cx1 = int(round(401 + (422.5 - 401) * (t / 79)))
        cy1 = int(round(294 + (509.0 - 294) * (t / 79)))
        
        cx2 = int(round(444 + (422.5 - 444) * (t / 79)))
        cy2 = int(round(724 + (509.0 - 724) * (t / 79)))
        
        F1_color = np.zeros((1024, 1024, 3), dtype=np.uint8)
        F1_mask = np.zeros((1024, 1024), dtype=bool)
        
        F2_color = np.zeros((1024, 1024, 3), dtype=np.uint8)
        F2_mask = np.zeros((1024, 1024), dtype=bool)
        
        # place 1
        y0_1, y1_1 = cy1 - 120, cy1 + 121
        x0_1, x1_1 = cx1 - 120, cx1 + 121
        F1_color[y0_1:y1_1, x0_1:x1_1][m1] = s1[m1]
        F1_mask[y0_1:y1_1, x0_1:x1_1] = m1
        
        # place 2
        y0_2, y1_2 = cy2 - 120, cy2 + 121
        x0_2, x1_2 = cx2 - 120, cx2 + 121
        F2_color[y0_2:y1_2, x0_2:x1_2][m2] = s2[m2]
        F2_mask[y0_2:y1_2, x0_2:x1_2] = m2
        
        both = F1_mask & F2_mask
        only1 = F1_mask & ~F2_mask
        only2 = F2_mask & ~F1_mask
        
        out = np.full((1024, 1024, 3), 255, dtype=np.uint8)
        
        out[both] = np.minimum(F1_color[both].astype(np.uint16) + F2_color[both].astype(np.uint16), 255).astype(np.uint8)
        out[only1] = F1_color[only1]
        out[only2] = F2_color[only2]
        
        # Convert BGR to RGB for imageio
        out_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        frames.append(out_rgb)
        
    imageio.mimwrite('/app/output/video.mp4', frames, fps=16, macro_block_size=None, codec='libx264', pixelformat='yuv420p')

if __name__ == '__main__':
    make_video()
