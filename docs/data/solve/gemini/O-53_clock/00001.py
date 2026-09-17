import cv2
import numpy as np
import os
import imageio

def get_rect_pts_shift(center_x, center_y, angle_deg, length, width, back_length):
    rad = np.deg2rad(angle_deg)
    dir_x = np.cos(rad)
    dir_y = np.sin(rad)
    norm_x = -np.sin(rad)
    norm_y = np.cos(rad)
    hw = width / 2.0
    bx = center_x - back_length * dir_x
    by = center_y - back_length * dir_y
    fx = center_x + length * dir_x
    fy = center_y + length * dir_y
    pts = [
        (bx + hw * norm_x, by + hw * norm_y),
        (fx + hw * norm_x, fy + hw * norm_y),
        (fx - hw * norm_x, fy - hw * norm_y),
        (bx - hw * norm_x, by - hw * norm_y)
    ]
    pts = [(int(round(x * 16)), int(round(y * 16))) for x, y in pts]
    return np.array([pts], dtype=np.int32)

def main():
    os.makedirs('/app/output', exist_ok=True)
    
    img = cv2.imread('/app/first_frame.png')
    
    hour_color = np.array([19, 69, 139])
    minute_color = np.array([60, 20, 220])
    
    diff_h = np.linalg.norm(img - hour_color, axis=2)
    diff_m = np.linalg.norm(img - minute_color, axis=2)
    
    hour_mask = (diff_h < 20).astype(np.uint8) * 255
    minute_mask = (diff_m < 20).astype(np.uint8) * 255
    
    kernel = np.ones((7,7), np.uint8)
    mask = cv2.dilate(cv2.bitwise_or(hour_mask, minute_mask), kernel, iterations=2)
    
    bg = img.copy()
    bg[mask > 0] = [255, 248, 240]
    
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', macro_block_size=None, pixelformat='yuv420p')
    
    num_frames = 120
    start_minutes = 604
    total_duration_minutes = 660
    
    for i in range(num_frames):
        if i == 0:
            frame = img.copy()
        else:
            frame = bg.copy()
            progress = i / (num_frames - 1)
            elapsed = progress * total_duration_minutes
            current_total_minutes = start_minutes + elapsed
            
            h_angle = 270 + current_total_minutes * 0.5
            m_angle = 270 + current_total_minutes * 6
            
            pts_h = get_rect_pts_shift(512, 512, h_angle, 206, 7.73, 11)
            cv2.fillPoly(frame, pts_h, (19, 69, 139), lineType=cv2.LINE_AA, shift=4)
            
            pts_m = get_rect_pts_shift(512, 512, m_angle, 287, 5.37, 11)
            cv2.fillPoly(frame, pts_m, (60, 20, 220), lineType=cv2.LINE_AA, shift=4)
            
            cv2.circle(frame, (512, 512), int(10.5 * 16), (0, 0, 0), -1, cv2.LINE_AA, shift=4)
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_rgb)
        
    writer.close()

if __name__ == '__main__':
    main()
