import cv2
import numpy as np
import os
import subprocess

def ease(t):
    # Smoothstep
    return t * t * (3.0 - 2.0 * t)

def main():
    # Load first frame
    image_path = '/app/first_frame.png'
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image at {image_path}")
    
    H_img, W_img, _ = image.shape
    bg_color = [255, 255, 255]
    
    # Extract circles
    diff = cv2.absdiff(image, np.full_like(image, bg_color))
    mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    circles = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        patch = image[y:y+h, x:x+w].copy()
        circles.append({
            'start_x': x,
            'start_y': y,
            'w': w,
            'h': h,
            'patch': patch
        })
        
    # Sort circles by width (circumference) descending
    circles.sort(key=lambda c: c['w'], reverse=True)
    
    # Calculate target positions
    gap = 20
    total_w = sum(c['w'] for c in circles) + gap * (len(circles) - 1)
    
    # Center the row
    current_x = (W_img - total_w) // 2
    
    for c in circles:
        c['end_x'] = current_x
        c['end_y'] = int(round(H_img / 2.0 - c['h'] / 2.0))
        current_x += c['w'] + gap
        
    # Animation parameters
    total_frames = 80
    fps = 16
    output_dir = '/app/output'
    os.makedirs(output_dir, exist_ok=True)
    frames_dir = os.path.join(output_dir, 'frames')
    os.makedirs(frames_dir, exist_ok=True)
    
    # Generate frames
    for f in range(total_frames):
        t = f / float(total_frames - 1)
        et = ease(t)
        
        # Start with clean background
        frame = np.full((H_img, W_img, 3), 255, dtype=np.uint8)
        
        # Draw circles, drawing smaller ones last (they are at the end of the sorted list)
        for c in circles:
            cur_x = int(round(c['start_x'] + (c['end_x'] - c['start_x']) * et))
            cur_y = int(round(c['start_y'] + (c['end_y'] - c['start_y']) * et))
            
            patch = c['patch']
            patch_mask = np.any(patch != 255, axis=-1)
            
            # Ensure we don't go out of bounds (though we shouldn't with 1024x1024)
            y1, y2 = cur_y, cur_y + c['h']
            x1, x2 = cur_x, cur_x + c['w']
            
            # In case of out of bounds, crop (just as a safety measure)
            # Actually, since W=1024 and we centered it, it won't go out of bounds.
            out_patch = frame[y1:y2, x1:x2]
            out_patch[patch_mask] = patch[patch_mask]
            frame[y1:y2, x1:x2] = out_patch
            
        frame_path = os.path.join(frames_dir, f'frame_{f:04d}.png')
        cv2.imwrite(frame_path, frame)
        
    # Create video
    video_path = os.path.join(output_dir, 'video.mp4')
    cmd = [
        'ffmpeg', '-y', '-framerate', str(fps),
        '-i', os.path.join(frames_dir, 'frame_%04d.png'),
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        video_path
    ]
    subprocess.run(cmd, check=True)
    print(f"Video saved to {video_path}")

if __name__ == '__main__':
    main()
