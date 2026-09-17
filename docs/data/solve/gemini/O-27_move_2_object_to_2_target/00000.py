import cv2
import numpy as np
import imageio
import os

def get_bbox(mask):
    y, x = np.where(mask)
    if len(y) == 0: return None
    return (x.min(), y.min(), x.max(), y.max())

def solve():
    img = cv2.imread('/app/first_frame.png')
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    bg_color = np.array([220, 220, 220])

    non_bg_mask = np.any(img_rgb != bg_color, axis=-1)

    # Distinguish Red and Yellow
    red_mask = non_bg_mask & (img_rgb[:,:,0] > img_rgb[:,:,1] + 20) & (img_rgb[:,:,0] > img_rgb[:,:,2] + 20)
    yellow_mask = non_bg_mask & (img_rgb[:,:,0] > img_rgb[:,:,2] + 20) & (img_rgb[:,:,1] > img_rgb[:,:,2] + 20)

    # Distinguish solid (object) from dashed (target)
    max_c = np.max(img_rgb, axis=-1)
    solid_mask = max_c > 80
    target_mask = max_c <= 80

    red_obj_mask = red_mask & solid_mask
    red_tgt_mask = red_mask & target_mask

    yellow_obj_mask = yellow_mask & solid_mask
    yellow_tgt_mask = yellow_mask & target_mask

    red_bbox = get_bbox(red_obj_mask)
    red_tgt_bbox = get_bbox(red_tgt_mask)
    
    yellow_bbox = get_bbox(yellow_obj_mask)
    yellow_tgt_bbox = get_bbox(yellow_tgt_mask)

    # Extract sub-images and exact masks for the objects
    r_sub = img_rgb[red_bbox[1]:red_bbox[3]+1, red_bbox[0]:red_bbox[2]+1]
    r_mask = red_obj_mask[red_bbox[1]:red_bbox[3]+1, red_bbox[0]:red_bbox[2]+1]
    
    y_sub = img_rgb[yellow_bbox[1]:yellow_bbox[3]+1, yellow_bbox[0]:yellow_bbox[2]+1]
    y_mask = yellow_obj_mask[yellow_bbox[1]:yellow_bbox[3]+1, yellow_bbox[0]:yellow_bbox[2]+1]

    clean_bg = img_rgb.copy()
    clean_bg[red_bbox[1]:red_bbox[3]+1, red_bbox[0]:red_bbox[2]+1][r_mask] = bg_color
    clean_bg[yellow_bbox[1]:yellow_bbox[3]+1, yellow_bbox[0]:yellow_bbox[2]+1][y_mask] = bg_color

    # Compute shifts based on bounding box centers
    red_total_dx = (red_tgt_bbox[0] + red_tgt_bbox[2])/2.0 - (red_bbox[0] + red_bbox[2])/2.0
    red_total_dy = (red_tgt_bbox[1] + red_tgt_bbox[3])/2.0 - (red_bbox[1] + red_bbox[3])/2.0

    yellow_total_dx = (yellow_tgt_bbox[0] + yellow_tgt_bbox[2])/2.0 - (yellow_bbox[0] + yellow_bbox[2])/2.0
    yellow_total_dy = (yellow_tgt_bbox[1] + yellow_tgt_bbox[3])/2.0 - (yellow_bbox[1] + yellow_bbox[3])/2.0

    frames = []
    num_frames = 35
    
    for i in range(num_frames):
        progress = i / (num_frames - 1)
        frame = clean_bg.copy()

        # Current shifts
        r_dx = int(np.round(red_total_dx * progress))
        r_dy = int(np.round(red_total_dy * progress))
        
        y_dx = int(np.round(yellow_total_dx * progress))
        y_dy = int(np.round(yellow_total_dy * progress))

        # Paste red
        rx1 = red_bbox[0] + r_dx
        ry1 = red_bbox[1] + r_dy
        rx2 = rx1 + (red_bbox[2] - red_bbox[0])
        ry2 = ry1 + (red_bbox[3] - red_bbox[1])
        frame[ry1:ry2+1, rx1:rx2+1][r_mask] = r_sub[r_mask]

        # Paste yellow
        yx1 = yellow_bbox[0] + y_dx
        yy1 = yellow_bbox[1] + y_dy
        yx2 = yx1 + (yellow_bbox[2] - yellow_bbox[0])
        yy2 = yy1 + (yellow_bbox[3] - yellow_bbox[1])
        frame[yy1:yy2+1, yx1:yx2+1][y_mask] = y_sub[y_mask]

        frames.append(frame)

    os.makedirs('/app/output', exist_ok=True)
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', pixelformat='yuv420p')
    for f in frames:
        writer.append_data(f)
    writer.close()

if __name__ == '__main__':
    solve()
