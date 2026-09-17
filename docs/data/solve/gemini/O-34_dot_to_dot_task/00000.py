import cv2
import numpy as np
import imageio
import math
import os

def main():
    img = cv2.imread('/app/first_frame.png')
    
    # Coordinates of the 8 dots (extracted from colored clusters)
    points = {
        1: (798, 459),
        2: (450, 508),
        3: (755, 666),
        4: (258, 291),
        5: (346, 866),
        6: (875, 229),
        7: (597, 257),
        8: (551, 843)
    }
    
    R = 46 # Radius of the dots mask to ensure line doesn't overwrite them
    
    # Precalculate segment lengths
    segments = []
    V_total = 0
    for i in range(1, 8):
        p1 = points[i]
        p2 = points[i+1]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        L = math.hypot(dx, dy)
        L_vis = max(0, L - 2*R)
        segments.append({
            'p1': p1, 'p2': p2, 'dx': dx, 'dy': dy, 'L': L, 'L_vis': L_vis
        })
        V_total += L_vis
    
    # Create mask to erase lines inside the dots' radius
    mask = np.ones(img.shape[:2], dtype=np.uint8) * 255
    for pt in points.values():
        cv2.circle(mask, pt, R, 0, -1, cv2.LINE_AA)
    mask_f = mask.astype(np.float32) / 255.0
    
    os.makedirs('/app/output', exist_ok=True)
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', pixelformat='yuv420p', macro_block_size=None)
    
    total_frames = 115
    for f in range(total_frames):
        vis_drawn = (f / (total_frames - 1)) * V_total
        
        if vis_drawn > 0:
            line_img = np.zeros_like(img)
            for seg in segments:
                if vis_drawn >= seg['L_vis'] - 1e-5:
                    cv2.line(line_img, seg['p1'], seg['p2'], (0, 0, 255), 5, cv2.LINE_AA)
                    vis_drawn -= seg['L_vis']
                else:
                    ratio = (R + vis_drawn) / seg['L']
                    end_x = seg['p1'][0] + seg['dx'] * ratio
                    end_y = seg['p1'][1] + seg['dy'] * ratio
                    cv2.line(line_img, seg['p1'], (int(round(end_x)), int(round(end_y))), (0, 0, 255), 5, cv2.LINE_AA)
                    break
                    
            # Blend lines with original image using the mask
            alpha_line = line_img[:, :, 2].astype(np.float32) / 255.0
            alpha = alpha_line * mask_f
            alpha_3d = np.expand_dims(alpha, axis=2)
            
            result = img.astype(np.float32) * (1 - alpha_3d) + np.array([0, 0, 255], dtype=np.float32) * alpha_3d
            result = np.clip(result, 0, 255).astype(np.uint8)
        else:
            result = img.copy()
            
        rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        writer.append_data(rgb)
        
    writer.close()

if __name__ == '__main__':
    main()
