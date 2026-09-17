import cv2
import numpy as np
import subprocess
import os

def create_video():
    img = cv2.imread('/app/first_frame.png')
    
    # The max value is 41.9%, its color in BGR is [0, 140, 255]
    color = np.array([0, 140, 255])
    mask = cv2.inRange(img, color, color)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if not contours:
        print("Error: Contour not found!")
        return
        
    c = max(contours, key=cv2.contourArea)
    points = c.squeeze()
    
    # Calculate cumulative distances
    # Ensure points are a 2D array of shape (N, 2)
    if points.ndim != 2:
        print("Error: Invalid contour shape", points.shape)
        return
        
    N = len(points)
    
    # We want to close the contour for drawing
    points = np.vstack([points, points[0]])
    N += 1
    
    diffs = np.diff(points, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    cum_dists = np.insert(np.cumsum(dists), 0, 0)
    total_length = cum_dists[-1]
    
    num_frames = 48
    fps = 16
    output_dir = '/tmp/frames'
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(num_frames):
        # Progress from 0.0 to 1.0
        progress = i / (num_frames - 1)
        target_dist = total_length * progress
        
        frame = img.copy()
        
        if progress > 0:
            idx = np.searchsorted(cum_dists, target_dist) - 1
            idx = max(0, min(idx, N - 2))
            
            # Points to draw fully
            pts_to_draw = points[:idx + 1].tolist()
            
            # Interpolated point
            if dists[idx] > 0:
                fraction = (target_dist - cum_dists[idx]) / dists[idx]
                fraction = max(0.0, min(fraction, 1.0))
                p_interp = points[idx] + fraction * (points[idx + 1] - points[idx])
                pts_to_draw.append(p_interp.astype(np.int32))
            
            pts_array = np.array(pts_to_draw, dtype=np.int32)
            cv2.polylines(frame, [pts_array], isClosed=False, color=(0, 0, 255), thickness=5, lineType=cv2.LINE_AA)
            
        cv2.imwrite(f'{output_dir}/frame_{i:04d}.png', frame)
        
    # Create video
    video_path = '/app/output/video.mp4'
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    
    cmd = [
        'ffmpeg', '-y', '-framerate', str(fps),
        '-i', f'{output_dir}/frame_%04d.png',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        video_path
    ]
    subprocess.run(cmd, check=True)
    print(f"Video saved to {video_path}")

if __name__ == '__main__':
    create_video()
