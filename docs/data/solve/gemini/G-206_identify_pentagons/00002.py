import cv2
import numpy as np
import os
import subprocess
import shutil

def main():
    os.makedirs('/app/output', exist_ok=True)
    temp_dir = '/app/temp_frames'
    os.makedirs(temp_dir, exist_ok=True)
    
    img = cv2.imread('/app/first_frame.png')
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    pentagon_contour = None
    pentagon_approx = None
    
    for cnt in contours:
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        area = cv2.contourArea(cnt)
        if area > 100:
            if len(approx) == 5:
                pentagon_contour = cnt
                pentagon_approx = approx
                break
                
    if pentagon_contour is not None:
        M = cv2.moments(pentagon_contour)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        
        # Max distance from center to a point in the pentagon
        distances = [np.linalg.norm(pt[0] - np.array([cX, cY])) for pt in pentagon_approx]
        # To encircle it, the radius should be slightly larger than the max distance
        target_radius = int(np.max(distances)) + 12 
        
        num_frames = 30
        for i in range(num_frames):
            frame = img.copy()
            
            # Linearly expand the radius
            current_radius = int(target_radius * (i / (num_frames - 1)))
            
            if current_radius > 0:
                cv2.circle(frame, (cX, cY), current_radius, (0, 0, 255), 5)
                
            cv2.imwrite(os.path.join(temp_dir, f"frame_{i:04d}.png"), frame)
            
        # Use ffmpeg to create the video
        subprocess.run([
            'ffmpeg', '-y', 
            '-framerate', '16', 
            '-i', os.path.join(temp_dir, 'frame_%04d.png'), 
            '-c:v', 'libx264', 
            '-pix_fmt', 'yuv420p', 
            '/app/output/video.mp4'
        ], check=True)
        
        # Clean up
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()
