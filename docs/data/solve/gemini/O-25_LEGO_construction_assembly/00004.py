import cv2
import numpy as np
import imageio
import os

def create_video():
    img = cv2.imread('/app/first_frame.png')
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 1. Detect red arrow
    mask_red = cv2.inRange(hsv, np.array([0, 150, 150]), np.array([10, 255, 255])) | \
               cv2.inRange(hsv, np.array([170, 150, 150]), np.array([180, 255, 255]))
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_red, connectivity=8)
    arrow_label = -1
    max_area = 0
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > max_area and stats[i, cv2.CC_STAT_AREA] > 500:
            max_area = stats[i, cv2.CC_STAT_AREA]
            arrow_label = i

    arrow_mask = (labels == arrow_label).astype(np.uint8) * 255
    kernel3 = np.ones((3,3), np.uint8)
    arrow_mask_dilated = cv2.dilate(arrow_mask, kernel3, iterations=1)

    # Find arrow start and end for translation
    pts = np.argwhere(arrow_mask > 0)
    left_tip = pts[np.argmin(pts[:, 1])]  # (y, x)
    right_tip = pts[np.argmax(pts[:, 1])] # (y, x)
    dx = right_tip[1] - left_tip[1]
    dy = right_tip[0] - left_tip[0]

    # 2. Extract green brick
    roi_x, roi_y, roi_w, roi_h = 90, 320, 140, 160
    roi = img[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w].copy()

    # Inpaint red arrow just on the brick ROI before extracting
    roi_arrow_mask = arrow_mask_dilated[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
    roi_inpainted = cv2.inpaint(roi, roi_arrow_mask, 3, cv2.INPAINT_TELEA)

    bg_color = np.median(roi_inpainted[:10, :10], axis=(0,1))
    diff = np.abs(roi_inpainted.astype(float) - bg_color)
    dist = np.sum(diff, axis=2)

    fg_mask = (dist > 30).astype(np.uint8) * 255
    
    num_labels_fg, labels_fg, stats_fg, centroids_fg = cv2.connectedComponentsWithStats(fg_mask, connectivity=8)
    brick_label = -1
    min_dist_to_center = 1000
    for i in range(1, num_labels_fg):
        if stats_fg[i, cv2.CC_STAT_AREA] > 1000:
            cx, cy = centroids_fg[i]
            d = (cx - roi_w/2)**2 + (cy - roi_h/2)**2
            if d < min_dist_to_center:
                min_dist_to_center = d
                brick_label = i

    brick_mask = (labels_fg == brick_label).astype(np.uint8) * 255
    kernel5 = np.ones((5,5), np.uint8)
    brick_mask = cv2.morphologyEx(brick_mask, cv2.MORPH_CLOSE, kernel5)

    # Extract brick with alpha
    b, g, r = cv2.split(roi_inpainted)
    brick_rgba = cv2.merge((b, g, r, brick_mask))
    y_idx, x_idx = np.where(brick_mask > 0)
    min_x, max_x = np.min(x_idx), np.max(x_idx)
    min_y, max_y = np.min(y_idx), np.max(y_idx)
    brick_cropped = brick_rgba[min_y:max_y+1, min_x:max_x+1]

    # Initial position of top-left corner of the cropped brick
    x_start = roi_x + min_x
    y_start = roi_y + min_y

    # 3. Create clean background
    bg = cv2.inpaint(img, arrow_mask_dilated, 3, cv2.INPAINT_TELEA)
    # Erase the exact brick mask
    brick_mask_dilated = cv2.dilate(brick_mask, kernel3, iterations=2)
    roi_bg = bg[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
    roi_bg[brick_mask_dilated == 255] = bg_color.astype(np.uint8)
    bg[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w] = roi_bg

    # 4. Generate frames
    frames = []
    # Frame 0 is the original frame
    frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    num_frames = 46
    brick_rgb = brick_cropped[:, :, :3]
    brick_alpha = brick_cropped[:, :, 3] / 255.0
    brick_h, brick_w = brick_cropped.shape[:2]

    for i in range(1, num_frames):
        t = (i - 1) / (num_frames - 2)
        # Smooth step: ease-in ease-out
        t_smooth = 3 * (t ** 2) - 2 * (t ** 3)
        
        x_curr = int(x_start + dx * t_smooth)
        y_curr = int(y_start + dy * t_smooth)

        # Make a copy of clean background
        frame = bg.copy()

        # Blend
        y1, y2 = y_curr, y_curr + brick_h
        x1, x2 = x_curr, x_curr + brick_w
        
        # Handle bounds
        if y1 < 0: y1 = 0
        if y2 > frame.shape[0]: y2 = frame.shape[0]
        if x1 < 0: x1 = 0
        if x2 > frame.shape[1]: x2 = frame.shape[1]

        by1 = 0 if y_curr >= 0 else -y_curr
        by2 = brick_h if y_curr + brick_h <= frame.shape[0] else frame.shape[0] - y_curr
        bx1 = 0 if x_curr >= 0 else -x_curr
        bx2 = brick_w if x_curr + brick_w <= frame.shape[1] else frame.shape[1] - x_curr

        if y2 > y1 and x2 > x1:
            bg_crop = frame[y1:y2, x1:x2]
            alpha_crop = brick_alpha[by1:by2, bx1:bx2, np.newaxis]
            rgb_crop = brick_rgb[by1:by2, bx1:bx2]
            blended = rgb_crop * alpha_crop + bg_crop * (1 - alpha_crop)
            frame[y1:y2, x1:x2] = blended.astype(np.uint8)
        
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # 5. Save video
    os.makedirs('/app/output', exist_ok=True)
    imageio.mimwrite('/app/output/video.mp4', frames, fps=16, macro_block_size=None, format='FFMPEG', codec='libx264', pixelformat='yuv420p')

if __name__ == '__main__':
    create_video()
