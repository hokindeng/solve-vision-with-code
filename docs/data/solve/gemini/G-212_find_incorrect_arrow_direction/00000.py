import cv2
import numpy as np
import os
import subprocess
import shutil

def find_outlier_arrow():
    img = cv2.imread('/app/first_frame.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_contours = []
    for c in contours:
        if cv2.contourArea(c) > 100:
            valid_contours.append(c)

    sum_x = 0
    sum_y = 0
    arrows_info = []

    for c in valid_contours:
        M = cv2.moments(c)
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            sum_x += cx
            sum_y += cy
            
            max_dist = 0
            furthest_pt = (cx, cy)
            for pt in c:
                px, py = pt[0]
                dist = (px - cx)**2 + (py - cy)**2
                if dist > max_dist:
                    max_dist = dist
                    furthest_pt = (px, py)
                    
            arrows_info.append({
                'contour': c,
                'cx': cx,
                'cy': cy,
                'tip_x': furthest_pt[0],
                'tip_y': furthest_pt[1]
            })

    center_x = sum_x / len(valid_contours)
    center_y = sum_y / len(valid_contours)

    cross_products = []
    for info in arrows_info:
        pos_x = info['cx'] - center_x
        pos_y = info['cy'] - center_y
        
        dir_x = info['tip_x'] - info['cx']
        dir_y = info['tip_y'] - info['cy']
        
        cross_p = pos_x * dir_y - pos_y * dir_x
        cross_products.append((cross_p, info))

    positive_count = sum(1 for cp, _ in cross_products if cp > 0)
    negative_count = sum(1 for cp, _ in cross_products if cp < 0)

    majority_is_positive = positive_count > negative_count

    outlier_info = None
    for cp, info in cross_products:
        if majority_is_positive and cp < 0:
            outlier_info = info
            break
        elif not majority_is_positive and cp > 0:
            outlier_info = info
            break

    return outlier_info

def generate_video():
    outlier = find_outlier_arrow()
    if not outlier:
        print("Could not find outlier arrow.")
        return

    x, y, w, h = cv2.boundingRect(outlier['contour'])
    cx_circle = x + w // 2
    cy_circle = y + h // 2
    radius = max(w, h) // 2 + 25

    frames_dir = '/tmp/frames_arrow'
    if os.path.exists(frames_dir):
        shutil.rmtree(frames_dir)
    os.makedirs(frames_dir, exist_ok=True)
    
    img_base = cv2.imread('/app/first_frame.png')

    num_frames = 48
    for i in range(num_frames):
        frame = img_base.copy()
        angle = i / (num_frames - 1) * 360.0
        
        if angle > 0:
            cv2.ellipse(frame, (cx_circle, cy_circle), (radius, radius), 0, -90, -90 + angle, (0, 0, 255), 6, cv2.LINE_AA)
            
        cv2.imwrite(os.path.join(frames_dir, f'frame_{i:04d}.png'), frame)

    os.makedirs('/app/output', exist_ok=True)
    cmd = [
        'ffmpeg', '-y',
        '-framerate', '16',
        '-i', os.path.join(frames_dir, 'frame_%04d.png'),
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '/app/output/video.mp4'
    ]
    subprocess.run(cmd, check=True)
    print("Video generated successfully.")

if __name__ == "__main__":
    generate_video()
