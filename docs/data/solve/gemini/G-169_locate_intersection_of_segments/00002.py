import cv2
import numpy as np
import os
import subprocess

def find_intersection(img):
    # Find the mask of the colored lines (anything not white)
    mask = np.any(img != [255, 255, 255], axis=-1).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img.shape[1] // 2, img.shape[0] // 2
        
    c = contours[0]
    hull = cv2.convexHull(c, returnPoints=False)
    defects = cv2.convexityDefects(c, hull)
    
    points = []
    if defects is not None:
        for i in range(defects.shape[0]):
            d = defects[i, 3]
            # 10000 in 1/256th pixel is ~39 pixels depth.
            # This robustly finds the 4 inner corners of the intersection X shape.
            if d > 10000:
                f = defects[i, 2]
                points.append(c[f][0])
                
    if len(points) == 4:
        center = np.mean(points, axis=0)
        return (int(center[0]), int(center[1]))
        
    return (img.shape[1]//2, img.shape[0]//2) # Fallback

def main():
    os.makedirs('/app/output', exist_ok=True)
    os.makedirs('/app/temp_frames', exist_ok=True)
    
    img = cv2.imread('/app/first_frame.png')
    if img is None:
        print("Error: Could not read /app/first_frame.png")
        return
        
    cx, cy = find_intersection(img)
    print(f"Intersection at: {cx}, {cy}")
    
    total_frames = 30
    fps = 16
    
    radius = 35
    color = (0, 0, 255) # Red in BGR
    thickness = 3
    
    start_draw_frame = 3
    end_draw_frame = 26
    
    for frame_idx in range(total_frames):
        frame = img.copy()
        
        if frame_idx >= start_draw_frame:
            # Calculate drawing progress (0.0 to 1.0)
            progress = (frame_idx - start_draw_frame) / (end_draw_frame - start_draw_frame)
            progress = min(max(progress, 0.0), 1.0)
            
            end_angle = int(360 * progress)
            
            if end_angle > 0:
                # Draw the red circle progressively. 
                # -90 starts drawing from the top.
                cv2.ellipse(frame, (cx, cy), (radius, radius), -90, 0, end_angle, color, thickness, cv2.LINE_AA)
                
        # Save frame
        cv2.imwrite(f'/app/temp_frames/frame_{frame_idx:04d}.png', frame)
        
    # Compile video with ffmpeg to exactly match the requested specs:
    # H.264, yuv420p, 1024x1024, 16 fps
    subprocess.run([
        'ffmpeg', '-y', 
        '-framerate', str(fps),
        '-i', '/app/temp_frames/frame_%04d.png',
        '-c:v', 'libx264', 
        '-pix_fmt', 'yuv420p',
        '/app/output/video.mp4'
    ], check=True)
    
    print("Video generated successfully.")

if __name__ == '__main__':
    main()
