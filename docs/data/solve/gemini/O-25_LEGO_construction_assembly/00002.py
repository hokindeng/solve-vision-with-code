import cv2
import numpy as np
import imageio
import os

def ease_in_out_cubic(t):
    return 4 * t * t * t if t < 0.5 else 1 - pow(-2 * t + 2, 3) / 2

def ease_in_cubic(t):
    return t * t * t

def generate_video():
    img = cv2.imread('/app/first_frame.png')
    bg_color = np.array([240, 245, 245])

    # Red arrow + white border extraction
    hsv_full = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    mask_red_full = cv2.inRange(hsv_full, lower_red1, upper_red1) | cv2.inRange(hsv_full, lower_red2, upper_red2)
    mask_red_full[600:, :] = 0

    kernel5 = np.ones((5,5), np.uint8)
    dilated_red = cv2.dilate(mask_red_full, kernel5)
    is_white = np.all(img == [255, 255, 255], axis=2).astype(np.uint8) * 255
    arrow_mask = mask_red_full | (is_white & dilated_red)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(arrow_mask, 8, cv2.CV_32S)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    arrow_mask_clean = (labels == largest_label).astype(np.uint8) * 255

    arrow_alpha = arrow_mask_clean.astype(np.float32) / 255.0
    arrow_rgb = img.astype(np.float32)

    # Blue brick extraction
    roi = img[280:530, 10:310].copy()
    diff = np.abs(roi.astype(int) - bg_color.astype(int))
    mask_non_bg = np.any(diff > 5, axis=-1).astype(np.uint8) * 255
    
    arrow_mask_clean_roi = arrow_mask_clean[280:530, 10:310]
    mask_brick = cv2.bitwise_and(mask_non_bg, cv2.bitwise_not(arrow_mask_clean_roi))

    bg_img = img.copy()
    roi_bg = bg_img[280:530, 10:310]
    for i in range(roi_bg.shape[0]):
        for j in range(roi_bg.shape[1]):
            if mask_brick[i, j] > 0:
                roi_bg[i, j] = bg_color

    # Reconstruct full alpha of blue brick
    mask_brick_filled = mask_brick.copy()
    for j in range(mask_brick.shape[1]):
        col = mask_brick[:, j]
        y_indices = np.where(col > 0)[0]
        if len(y_indices) >= 2:
            min_y, max_y = y_indices[0], y_indices[-1]
            for i in range(min_y, max_y + 1):
                if arrow_mask_clean_roi[i, j] > 0:
                    mask_brick_filled[i, j] = 255

    inpainted_roi = cv2.inpaint(roi, arrow_mask_clean_roi, 3, cv2.INPAINT_TELEA)

    brick_rgba = np.zeros((roi.shape[0], roi.shape[1], 4), dtype=np.float32)
    brick_rgba[..., :3] = inpainted_roi
    brick_rgba[..., 3] = mask_brick_filled / 255.0

    # Animation
    target_dx = 624 - 160
    target_dy = 582 - 419
    snap_height = 40
    
    frames = []
    num_frames = 46
    p1_frames = 32
    p2_frames = 8
    
    for f in range(num_frames):
        if f == 0:
            dx, dy = 0, 0
        elif f < p1_frames:
            t = f / p1_frames
            eased_t = ease_in_out_cubic(t)
            dx = int(target_dx * eased_t)
            dy = int((target_dy - snap_height) * eased_t)
        elif f < p1_frames + p2_frames:
            t = (f - p1_frames) / p2_frames
            eased_t = ease_in_cubic(t)
            dx = target_dx
            dy = int((target_dy - snap_height) + snap_height * eased_t)
        else:
            dx = target_dx
            dy = target_dy
            
        frame = bg_img.copy().astype(np.float32)
        
        y1, y2 = 280 + dy, 280 + dy + roi.shape[0]
        x1, x2 = 10 + dx, 10 + dx + roi.shape[1]
        
        y1_c = max(0, y1)
        y2_c = min(frame.shape[0], y2)
        x1_c = max(0, x1)
        x2_c = min(frame.shape[1], x2)
        
        if y1_c < y2_c and x1_c < x2_c:
            brick_y1 = y1_c - y1
            brick_y2 = roi.shape[0] - (y2 - y2_c)
            brick_x1 = x1_c - x1
            brick_x2 = roi.shape[1] - (x2 - x2_c)
            
            alpha = brick_rgba[brick_y1:brick_y2, brick_x1:brick_x2, 3:4]
            frame[y1_c:y2_c, x1_c:x2_c, :3] = alpha * brick_rgba[brick_y1:brick_y2, brick_x1:brick_x2, :3] + (1 - alpha) * frame[y1_c:y2_c, x1_c:x2_c, :3]
            
        frame = arrow_alpha[:, :, None] * arrow_rgb + (1 - arrow_alpha[:, :, None]) * frame
            
        frames.append(cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_BGR2RGB))
        
    os.makedirs('/app/output', exist_ok=True)
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', pixelformat='yuv420p', macro_block_size=None)
    for frame in frames:
        writer.append_data(frame)
    writer.close()

if __name__ == '__main__':
    generate_video()
