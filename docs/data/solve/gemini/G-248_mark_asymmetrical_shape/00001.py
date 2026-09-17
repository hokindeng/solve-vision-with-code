import cv2
import numpy as np
import imageio
import os

def get_symmetry_score(c, mask_shape):
    M = cv2.moments(c)
    if M['m00'] == 0: return 0
    cx = M['m10'] / M['m00']
    cy = M['m01'] / M['m00']
    
    mu20 = M['mu20'] / M['m00']
    mu02 = M['mu02'] / M['m00']
    mu11 = M['mu11'] / M['m00']
    
    theta = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
    deg = np.degrees(-theta)
    
    mask = np.zeros(mask_shape, dtype=np.uint8)
    cv2.drawContours(mask, [c], -1, 255, -1)
    
    R180 = cv2.getRotationMatrix2D((cx, cy), 180, 1.0)
    rot180 = cv2.warpAffine(mask, R180, (mask_shape[1], mask_shape[0]))
    diff_180 = np.sum(cv2.absdiff(mask, rot180)) / 255
    
    R = cv2.getRotationMatrix2D((cx, cy), deg, 1.0)
    rotated = cv2.warpAffine(mask, R, (mask_shape[1], mask_shape[0]))
    
    x, y, w, h = cv2.boundingRect(rotated)
    roi = rotated[y:y+h, x:x+w]
    
    diff_x = np.sum(cv2.absdiff(roi, cv2.flip(roi, 0))) / 255
    diff_y = np.sum(cv2.absdiff(roi, cv2.flip(roi, 1))) / 255
    
    return min(diff_180, diff_x, diff_y)

def main():
    img = cv2.imread('/app/first_frame.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best_c = None
    max_score = -1
    for c in contours:
        # ignore tiny noise
        if cv2.contourArea(c) < 100: continue
        score = get_symmetry_score(c, gray.shape)
        if score > max_score:
            max_score = score
            best_c = c
            
    (cx, cy), radius = cv2.minEnclosingCircle(best_c)
    center = (int(cx), int(cy))
    radius = int(radius) + 15
    
    os.makedirs('/app/output', exist_ok=True)
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', macro_block_size=None, pixelformat='yuv420p')
    
    num_frames = 16
    for i in range(num_frames):
        frame = img.copy()
        if i > 0:
            end_angle = -90 + int(360 * i / (num_frames - 1))
            cv2.ellipse(frame, center, (radius, radius), 0, -90, end_angle, (0, 0, 255), 5, cv2.LINE_AA)
            
        writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
    writer.close()

if __name__ == '__main__':
    main()
