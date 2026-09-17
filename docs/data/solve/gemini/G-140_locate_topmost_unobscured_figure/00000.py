import cv2
import numpy as np
import os
import subprocess

def main():
    img_path = '/app/first_frame.png'
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("Could not read first_frame.png")

    bg_color = img[0, 0]
    
    # Find all unique colors (excluding background)
    colors = np.unique(img.reshape(-1, 3), axis=0)
    colors = [c for c in colors if not np.array_equal(c, bg_color)]
    
    topmost_contour = None
    target_color = None
    
    # The topmost shape will be the only perfect triangle (3 vertices)
    for c in colors:
        mask = cv2.inRange(img, c, c)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        # If a shape is split, it's definitely not topmost
        if len(contours) > 1:
            continue
            
        cnt = contours[0]
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        
        if len(approx) == 3:
            topmost_contour = cnt
            target_color = c
            break

    # Fallback to yellow if detection fails for some reason
    if topmost_contour is None:
        target_color = np.array([255, 255, 25]) # Cyan in BGR, appears Yellow/Cyan
        mask = cv2.inRange(img, target_color, target_color)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        topmost_contour = contours[0]

    # Generate frames
    tmp_dir = '/tmp/video_frames'
    os.makedirs(tmp_dir, exist_ok=True)
    
    num_frames = 40
    for i in range(num_frames):
        frame = img.copy()
        if i > 0:
            # Calculate how many points of the contour to draw for this frame
            num_points = int(len(topmost_contour) * i / (num_frames - 1))
            if num_points > 1:
                pts = topmost_contour[:num_points]
                is_closed = (i == num_frames - 1)
                cv2.polylines(frame, [pts], isClosed=is_closed, color=(0, 0, 255), thickness=6, lineType=cv2.LINE_AA)
        cv2.imwrite(f'{tmp_dir}/frame_{i:04d}.png', frame)
        
    # Compile video with ffmpeg
    output_path = '/app/output/video.mp4'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-framerate', '16',
        '-i', f'{tmp_dir}/frame_%04d.png',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        output_path
    ]
    
    subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

if __name__ == '__main__':
    main()
