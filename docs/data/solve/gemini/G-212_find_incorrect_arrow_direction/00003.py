import cv2
import numpy as np
import math
import imageio
import os

def generate_video():
    img = cv2.imread('/app/first_frame.png')
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    arrows = []
    for cnt in contours:
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            area = cv2.contourArea(cnt)
            if area > 100:
                x, y, w, h = cv2.boundingRect(cnt)
                center_x = x + w / 2.0
                center_y = y + h / 2.0
                
                furthest_pt = None
                max_dist = -1
                for pt in cnt:
                    px, py = pt[0]
                    dist = (px - cX)**2 + (py - cY)**2
                    if dist > max_dist:
                        max_dist = dist
                        furthest_pt = (px, py)
                        
                dir_x = furthest_pt[0] - cX
                dir_y = furthest_pt[1] - cY
                dir_angle = math.degrees(math.atan2(dir_y, dir_x)) % 360
                
                arrows.append({
                    'cnt': cnt, 'cx': cX, 'cy': cY, 'area': area,
                    'bx': center_x, 'by': center_y,
                    'dir_angle': dir_angle,
                    'w': w, 'h': h
                })

    global_cx = sum(a['cx'] for a in arrows) / len(arrows)
    global_cy = sum(a['cy'] for a in arrows) / len(arrows)

    for a in arrows:
        vx = a['cx'] - global_cx
        vy = a['cy'] - global_cy
        pos_angle = math.degrees(math.atan2(vy, vx)) % 360
        a['pos_angle'] = pos_angle
        a['diff'] = (a['dir_angle'] - pos_angle) % 360

    diffs = [a['diff'] for a in arrows]
    cos_sum = sum(math.cos(math.radians(d)) for d in diffs)
    sin_sum = sum(math.sin(math.radians(d)) for d in diffs)
    avg_diff = math.degrees(math.atan2(sin_sum, cos_sum)) % 360

    outlier = None
    max_diff_dist = -1
    for a in arrows:
        dist = abs(a['diff'] - avg_diff)
        dist = min(dist, 360 - dist)
        if dist > max_diff_dist:
            max_diff_dist = dist
            outlier = a

    center = (int(outlier['bx']), int(outlier['by']))
    # A bit of padding for the radius
    radius = int(max(outlier['w'], outlier['h']) / 2 + 30)
    
    os.makedirs('/app/output', exist_ok=True)
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', pixelformat='yuv420p', macro_block_size=None)
    
    total_frames = 48
    
    for i in range(total_frames):
        frame = img.copy()
        
        progress = i / (total_frames - 1) if total_frames > 1 else 1.0
        current_angle = int(progress * 360)
        
        if current_angle > 0:
            cv2.ellipse(frame, center, (radius, radius), 0, -90, -90 + current_angle, (0, 0, 255), 8)
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_rgb)
        
    writer.close()

if __name__ == '__main__':
    generate_video()
