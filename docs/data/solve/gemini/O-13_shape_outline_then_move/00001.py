import cv2
import numpy as np
import os
import subprocess

def main():
    img_path = '/app/first_frame.png'
    out_dir = '/app/output'
    os.makedirs(out_dir, exist_ok=True)
    out_path_raw = os.path.join(out_dir, 'video_raw.mp4')
    out_path = os.path.join(out_dir, 'video.mp4')

    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read {img_path}")

    # Ensure shape is 1024x1024
    if img.shape[:2] != (1024, 1024):
        img = cv2.resize(img, (1024, 1024))

    # Crescent D bounding box
    d_x, d_y = 108, 602
    d_w, d_h = 163, 161
    crescent_D = img[d_y:d_y+d_h, d_x:d_x+d_w].copy()

    # Create mask for drawing
    mask = np.any(crescent_D != [255, 255, 255], axis=-1)

    # Question marks bounding boxes
    qm1_box = (450, 650, 50, 50)  # x, y, w, h
    qm2_box = (740, 650, 50, 50)

    qm1_patch = img[qm1_box[1]:qm1_box[1]+qm1_box[3], qm1_box[0]:qm1_box[0]+qm1_box[2]].copy()
    qm2_patch = img[qm2_box[1]:qm2_box[1]+qm2_box[3], qm2_box[0]:qm2_box[0]+qm2_box[2]].copy()

    # Create background by erasing QMs
    bg = img.copy()
    bg[qm1_box[1]:qm1_box[1]+qm1_box[3], qm1_box[0]:qm1_box[0]+qm1_box[2]] = 255
    bg[qm2_box[1]:qm2_box[1]+qm2_box[3], qm2_box[0]:qm2_box[0]+qm2_box[2]] = 255

    # Target positions
    e_x, e_y = 397, 603
    f_x, f_y = 685, 663

    # Video setup
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path_raw, fourcc, 16.0, (1024, 1024))

    for t in range(64):
        frame = bg.copy()
        
        if t <= 31:
            progress = t / 31.0
            
            # Slide D to E
            cur_x = int(round(d_x + progress * (e_x - d_x)))
            cur_y = int(round(d_y + progress * (e_y - d_y)))
            
            # Draw fading QM1
            alpha1 = max(0.0, 1.0 - progress)
            white_patch = np.full_like(qm1_patch, 255)
            qm1_current = cv2.addWeighted(qm1_patch, alpha1, white_patch, 1 - alpha1, 0)
            frame[qm1_box[1]:qm1_box[1]+qm1_box[3], qm1_box[0]:qm1_box[0]+qm1_box[2]] = qm1_current
            
            # Draw static QM2
            frame[qm2_box[1]:qm2_box[1]+qm2_box[3], qm2_box[0]:qm2_box[0]+qm2_box[2]] = qm2_patch
            
            # Draw sliding shape (D -> E)
            roi = frame[cur_y:cur_y+d_h, cur_x:cur_x+d_w]
            np.copyto(roi, crescent_D, where=mask[:, :, None])
            
        else:
            progress = (t - 32) / 31.0 if t < 63 else 1.0
            
            # Slide E to F
            cur_x = int(round(e_x + progress * (f_x - e_x)))
            cur_y = int(round(e_y + progress * (f_y - e_y)))
            
            # Draw static E
            roi_e = frame[e_y:e_y+d_h, e_x:e_x+d_w]
            np.copyto(roi_e, crescent_D, where=mask[:, :, None])
            
            # Draw fading QM2
            alpha2 = max(0.0, 1.0 - progress)
            white_patch = np.full_like(qm2_patch, 255)
            qm2_current = cv2.addWeighted(qm2_patch, alpha2, white_patch, 1 - alpha2, 0)
            frame[qm2_box[1]:qm2_box[1]+qm2_box[3], qm2_box[0]:qm2_box[0]+qm2_box[2]] = qm2_current
            
            # Draw sliding shape (E -> F)
            roi_f = frame[cur_y:cur_y+d_h, cur_x:cur_x+d_w]
            np.copyto(roi_f, crescent_D, where=mask[:, :, None])
            
        out.write(frame)

    out.release()
    
    # Convert to H.264
    subprocess.run([
        'ffmpeg', '-y', '-i', out_path_raw, 
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-r', '16', 
        out_path
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    os.remove(out_path_raw)
    print(f"Video saved to {out_path}")

if __name__ == '__main__':
    main()
