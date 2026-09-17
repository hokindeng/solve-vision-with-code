import cv2
import numpy as np
import imageio
import os

def solve():
    # Ensure output directory exists
    os.makedirs('/app/output', exist_ok=True)
    
    # Read the original image
    img_bgr = cv2.imread('/app/first_frame.png')
    if img_bgr is None:
        raise FileNotFoundError("Could not find /app/first_frame.png")
        
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Dot centers computed from contours
    # 1: 512, 207 (91x91) -> center (557.5, 252.5)
    # 2: 102, 825 (91x91) -> center (147.5, 870.5)
    # 3: 550, 642 (91x91) -> center (595.5, 687.5)
    # 4: 312, 357 (91x91) -> center (357.5, 402.5)
    # 5: 723, 236 (91x91) -> center (768.5, 281.5)
    
    centers = [
        (557.5, 252.5),
        (147.5, 870.5),
        (595.5, 687.5),
        (357.5, 402.5),
        (768.5, 281.5)
    ]
    
    # Create mask for dots
    mask = np.zeros((1024, 1024), dtype=np.uint8)
    for cx, cy in centers:
        cv2.circle(mask, (int(round(cx)), int(round(cy))), 46, 255, -1)
        
    total_frames = 70
    writer = imageio.get_writer('/app/output/video.mp4', 
                                fps=16, 
                                codec='libx264', 
                                pixelformat='yuv420p', 
                                macro_block_size=None)
                                
    for f in range(total_frames):
        img_copy = img_rgb.copy()
        
        if f > 0:
            progress = (f / (total_frames - 1)) * 4.0
            
            for i in range(4):
                if progress > i:
                    p1 = np.array(centers[i])
                    p2 = np.array(centers[i+1])
                    
                    if progress >= i + 1:
                        # Line fully drawn
                        end_pt = p2
                    else:
                        # Line partially drawn
                        t = progress - i
                        end_pt = p1 + t * (p2 - p1)
                    
                    # Draw the line
                    cv2.line(img_copy, 
                             (int(round(p1[0])), int(round(p1[1]))), 
                             (int(round(end_pt[0])), int(round(end_pt[1]))), 
                             (255, 0, 0), # Red in RGB
                             thickness=6, 
                             lineType=cv2.LINE_AA)
                             
        # Restore the dots so the lines appear to go "behind" them
        np.copyto(img_copy, img_rgb, where=(mask[:,:,None] == 255))
        
        writer.append_data(img_copy)
        
    writer.close()

if __name__ == '__main__':
    solve()
