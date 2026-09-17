import cv2
import numpy as np
import imageio
import os

def main():
    os.makedirs('/app/output', exist_ok=True)
    img = cv2.imread('/app/first_frame.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_iou = 1.0
    asym_contour = None

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w*h < 100: continue
        
        mask = np.zeros_like(thresh)
        cv2.drawContours(mask, [c], -1, 255, -1)
        
        M = cv2.moments(mask)
        if M['m00'] == 0: continue
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        
        pts = np.where(mask > 0)
        max_dy = max(abs(np.max(pts[0]) - cy), abs(cy - np.min(pts[0])))
        max_dx = max(abs(np.max(pts[1]) - cx), abs(cx - np.min(pts[1])))
        R = int(max(max_dx, max_dy)) + 5
        
        roi = np.zeros((2*R, 2*R), dtype=np.uint8)
        
        y0, y1 = cy - R, cy + R
        x0, x1 = cx - R, cx + R
        
        h_img, w_img = mask.shape
        r_y0 = max(0, -y0)
        r_x0 = max(0, -x0)
        r_y1 = 2*R - max(0, y1 - h_img)
        r_x1 = 2*R - max(0, x1 - w_img)
        
        m_y0 = max(0, y0)
        m_x0 = max(0, x0)
        m_y1 = min(h_img, y1)
        m_x1 = min(w_img, x1)
        
        roi[r_y0:r_y1, r_x0:r_x1] = mask[m_y0:m_y1, m_x0:m_x1]
        
        max_iou_c = 0
        for angle in range(0, 180, 2):
            rot_M = cv2.getRotationMatrix2D((int(R), int(R)), float(angle), 1.0)
            rotated = cv2.warpAffine(roi, rot_M, (2*R, 2*R))
            
            flipped = cv2.flip(rotated, 1)
            
            intersection = cv2.bitwise_and(rotated, flipped)
            union = cv2.bitwise_or(rotated, flipped)
            iou = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0
            if iou > max_iou_c:
                max_iou_c = iou
                
        if max_iou_c < min_iou:
            min_iou = max_iou_c
            asym_contour = c

    (x,y), radius = cv2.minEnclosingCircle(asym_contour)
    center = (int(x), int(y))
    axes = (int(radius) + 15, int(radius) + 15) # Add 15px padding
    
    frames = []
    for i in range(16):
        frame = img.copy()
        if i > 0:
            # We want endAngle to go from 0 to 360 smoothly over the frames.
            # At i=0, arc is 0. At i=15, arc is 360.
            end_angle = -90 + (360 * i / 15)
            cv2.ellipse(frame, center, axes, 0, -90, end_angle, (0, 0, 255), 5, cv2.LINE_AA)
        
        # imageio expects RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', macro_block_size=None, pixelformat='yuv420p')
    for frame in frames:
        writer.append_data(frame)
    writer.close()

if __name__ == '__main__':
    main()
