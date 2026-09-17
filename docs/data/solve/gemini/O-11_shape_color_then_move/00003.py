import cv2
import numpy as np
import imageio
import os

def generate_video():
    os.makedirs('/app/output', exist_ok=True)

    base_img = cv2.imread('/app/first_frame.png')
    base_img = cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB)

    # Extract D ROI
    x_d, y_d, w_d, h_d = 115, 602, 161, 161
    d_roi = base_img[y_d:y_d+h_d, x_d:x_d+w_d]

    # Red-ish (blue-ish actually) color in RGB
    old_c = np.array([38, 63, 191], dtype=np.uint8)
    # Green color in RGB
    new_c = np.array([95, 191, 111], dtype=np.uint8)

    is_color = (d_roi[:, :, 0] == 38) & (d_roi[:, :, 1] == 63) & (d_roi[:, :, 2] == 191)

    x_e, y_e = 402, 602
    x_f = 689

    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', pixelformat='yuv420p')

    for frame in range(60):
        img = base_img.copy()
        
        if frame == 0:
            pass # Keep it exactly as base_img
        elif frame < 30:
            # Phase 1: Recolor at E
            t = frame / 29.0
            curr_c = old_c.astype(np.float32) * (1 - t) + new_c.astype(np.float32) * t
            curr_c = curr_c.astype(np.uint8)
            
            e_roi = d_roi.copy()
            e_roi[is_color] = curr_c
            
            img[y_e:y_e+h_d, x_e:x_e+w_d] = e_roi
            
        else:
            # Phase 2: Move down F
            e_roi = d_roi.copy()
            e_roi[is_color] = new_c
            
            img[y_e:y_e+h_d, x_e:x_e+w_d] = e_roi
            
            t = (frame - 30) / 29.0
            y_f = int(602 + 40 * t)
            
            f_roi = e_roi.copy()
            
            img[y_f:y_f+h_d, x_f:x_f+w_d] = f_roi
            
        writer.append_data(img)

    writer.close()

if __name__ == '__main__':
    generate_video()
