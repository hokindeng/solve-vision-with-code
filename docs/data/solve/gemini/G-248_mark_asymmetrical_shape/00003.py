import cv2
import numpy as np
import imageio
import os

def get_symmetry_score(contour):
    x, y, w, h = cv2.boundingRect(contour)
    mask = np.zeros((h + 100, w + 100), dtype=np.uint8)
    shifted_c = contour - [x - 50, y - 50]
    cv2.drawContours(mask, [shifted_c], -1, 255, -1)

    M = cv2.moments(mask)
    if M["m00"] == 0:
        return 0
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]

    min_diff = float('inf')
    for angle in range(0, 180, 2):
        rot_mat = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        rotated = cv2.warpAffine(mask, rot_mat, (mask.shape[1], mask.shape[0]))
        flipped = cv2.flip(rotated, 1)
        intersection = cv2.bitwise_and(rotated, flipped)
        union = cv2.bitwise_or(rotated, flipped)
        union_sum = np.sum(union)
        if union_sum == 0:
            continue
        iou = np.sum(intersection) / union_sum
        if 1 - iou < min_diff:
            min_diff = 1 - iou

    return 1 - min_diff

def main():
    os.makedirs('/app/output', exist_ok=True)
    img = cv2.imread('/app/first_frame.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out too small contours
    contours = [c for c in contours if cv2.contourArea(c) > 100]

    # Find the most asymmetric contour (lowest symmetry score)
    scores = [get_symmetry_score(c) for c in contours]
    asym_idx = np.argmin(scores)
    target_contour = contours[asym_idx]

    # Find center and radius for the circle
    x, y, w, h = cv2.boundingRect(target_contour)
    center_x = x + w // 2
    center_y = y + h // 2
    
    # Calculate exact distance to farthest point in contour
    max_dist = 0
    for pt in target_contour:
        px, py = pt[0]
        dist = np.sqrt((px - center_x)**2 + (py - center_y)**2)
        if dist > max_dist:
            max_dist = dist
            
    radius = int(max_dist + 15)

    writer = imageio.get_writer(
        '/app/output/video.mp4', 
        fps=16, 
        codec='libx264', 
        pixelformat='yuv420p',
        macro_block_size=None, 
        quality=8
    )
    
    frames_count = 16
    for i in range(frames_count):
        frame = img.copy()
        angle = int(360 * i / (frames_count - 1))
        if angle > 0:
            cv2.ellipse(frame, (center_x, center_y), (radius, radius), 0, 0, angle, (0, 0, 255), 5, cv2.LINE_AA)
        writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
    writer.close()

if __name__ == '__main__':
    main()
