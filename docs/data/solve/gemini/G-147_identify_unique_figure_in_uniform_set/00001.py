import cv2
import numpy as np
import os
import subprocess

def get_shape_features(c):
    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)
    if perimeter == 0:
        return np.array([0, 0, 0])
    circularity = 4 * np.pi * area / (perimeter * perimeter)
    x, y, w, h = cv2.boundingRect(c)
    aspect_ratio = w / float(h)
    return np.array([area, circularity, aspect_ratio])

def generate_video():
    img = cv2.imread('/app/first_frame.png')
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter out tiny noise contours
    contours = [c for c in contours if cv2.contourArea(c) > 100]
    
    features = np.array([get_shape_features(c) for c in contours])
    
    # Normalize features
    features_norm = features.copy()
    for i in range(features.shape[1]):
        col = features[:, i]
        ptp = np.ptp(col)
        if ptp > 0:
            features_norm[:, i] = (col - np.min(col)) / ptp
            
    # Find the outlier
    distances = np.zeros(len(features))
    for i in range(len(features)):
        dist_to_others = 0
        for j in range(len(features)):
            if i != j:
                dist_to_others += np.linalg.norm(features_norm[i] - features_norm[j])
        distances[i] = dist_to_others
        
    unique_idx = np.argmax(distances)
        
    c = contours[unique_idx]
    M = cv2.moments(c)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        x, y, w, h = cv2.boundingRect(c)
        cX = x + w // 2
        cY = y + h // 2
        
    center = (cX, cY)
    
    # Calculate radius based on the bounding box to tightly fit or comfortably circle it
    x, y, w, h = cv2.boundingRect(c)
    max_dim = max(w, h)
    radius = int(max_dim / 2) + 20
    
    thickness = 8
    color = (0, 0, 255) # Red in BGR
    
    os.makedirs('/app/output', exist_ok=True)
    out_path = '/app/output/video.mp4'
    temp_path = '/app/output/video_temp.mp4'
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, 16.0, (1024, 1024))
    
    total_frames = 60
    start_draw_frame = 5
    end_draw_frame = 50
    
    for frame_idx in range(total_frames):
        frame = img.copy()
        
        if frame_idx >= start_draw_frame:
            if frame_idx >= end_draw_frame:
                end_angle = 360
            else:
                progress = (frame_idx - start_draw_frame) / (end_draw_frame - start_draw_frame)
                # Apply easing out
                progress = 1 - (1 - progress) ** 2
                end_angle = progress * 360
                
            cv2.ellipse(frame, center, (radius, radius), -90, 0, end_angle, color, thickness, cv2.LINE_AA)
            
        out.write(frame)
        
    out.release()
    
    subprocess.run([
        'ffmpeg', '-y', '-i', out_path,
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        temp_path
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.rename(temp_path, out_path)

if __name__ == '__main__':
    generate_video()
