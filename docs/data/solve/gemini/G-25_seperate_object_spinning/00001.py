import cv2
import numpy as np
import imageio
import os

def solve():
    img = cv2.imread('/app/first_frame.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    solid_mask = np.zeros_like(thresh)
    for c in contours:
        if cv2.contourArea(c) > 1000:
            cv2.drawContours(solid_mask, [c], -1, 255, -1)

    # Background: original image minus the solid objects
    bg = img.copy()
    bg[solid_mask > 0] = (255, 255, 255)

    solids = [
        (147, 548, 197, 197),
        (42, 353, 109, 110),
        (162, 278, 260, 260)
    ]
    angles = [43.0, -14.0, 18.0]
    dxs = [560.0, 508.0, 514.0]

    # Extract objects as RGBA
    objects_rgba = []
    centers = []
    for i in range(3):
        x, y, w, h = solids[i]
        cx, cy = x + w/2.0, y + h/2.0
        centers.append((cx, cy))
        
        obj_mask = np.zeros_like(thresh)
        cv2.rectangle(obj_mask, (x, y), (x+w, y+h), 255, -1)
        obj_mask = cv2.bitwise_and(thresh, obj_mask)
        
        rgba = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)
        rgba[:, :, :3] = img
        rgba[:, :, 3] = obj_mask
        objects_rgba.append(rgba)

    os.makedirs('/app/output', exist_ok=True)
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', pixelformat='yuv420p')

    for f in range(48):
        frame = bg.copy()
        
        for i in range(3):
            # Calculate current angle and dx
            if f <= 23:
                cur_angle = angles[i] * f / 23.0
                cur_dx = 0.0
            else:
                cur_angle = angles[i]
                cur_dx = dxs[i] * (f - 23) / 24.0
                
            M = cv2.getRotationMatrix2D(centers[i], cur_angle, 1.0)
            M[0, 2] += cur_dx
            
            rotated = cv2.warpAffine(objects_rgba[i], M, (img.shape[1], img.shape[0]), flags=cv2.INTER_NEAREST)
            
            alpha = rotated[:, :, 3] > 0
            frame[alpha] = rotated[alpha, :3]
            
        writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
    writer.close()

if __name__ == '__main__':
    solve()
