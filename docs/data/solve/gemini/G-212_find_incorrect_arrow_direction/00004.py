import cv2
import numpy as np
import math
import os
import imageio

def solve():
    img = cv2.imread('/app/first_frame.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    arrows = []
    for i, cnt in enumerate(contours):
        M = cv2.moments(cnt)
        if M['m00'] > 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            arrows.append({"id": i, "cx": cx, "cy": cy, "cnt": cnt})

    avg_cx = sum(a['cx'] for a in arrows) / len(arrows)
    avg_cy = sum(a['cy'] for a in arrows) / len(arrows)

    for a in arrows:
        angle_from_center = math.degrees(math.atan2(a['cy'] - avg_cy, a['cx'] - avg_cx))
        
        epsilon = 0.02 * cv2.arcLength(a['cnt'], True)
        approx = cv2.approxPolyDP(a['cnt'], epsilon, True)
        max_dist = 0
        tip = None
        for pt in approx:
            px, py = pt[0]
            dist = math.hypot(px - a['cx'], py - a['cy'])
            if dist > max_dist:
                max_dist = dist
                tip = (px, py)
        
        arrow_angle = math.degrees(math.atan2(tip[1] - a['cy'], tip[0] - a['cx']))
        rel_angle = (arrow_angle - angle_from_center) % 360
        a['rel_angle'] = rel_angle

    rel_angles = [a['rel_angle'] for a in arrows]
    median_angle = np.median(rel_angles)
    
    def angle_diff(a1, a2):
        d = abs(a1 - a2) % 360
        return d if d < 180 else 360 - d
        
    odd_arrow = max(arrows, key=lambda a: angle_diff(a['rel_angle'], median_angle))
    
    (ccx, ccy), radius = cv2.minEnclosingCircle(odd_arrow['cnt'])
    center = (int(ccx), int(ccy))
    target_radius = int(radius + 15)

    out_path = '/app/output/video.mp4'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    fps = 16
    duration = 3.0
    total_frames = int(fps * duration)
    
    writer = imageio.get_writer(out_path, fps=fps, codec='libx264', pixelformat='yuv420p', macro_block_size=None, quality=10)
    
    for i in range(total_frames):
        frame = img.copy()
        progress = i / (total_frames - 1)
        end_angle = progress * 360
        
        if end_angle > 0:
            cv2.ellipse(frame, center, (target_radius, target_radius), -90, 0, end_angle, (0, 0, 255), 6, cv2.LINE_AA)
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_rgb)
        
    writer.close()

if __name__ == '__main__':
    solve()
