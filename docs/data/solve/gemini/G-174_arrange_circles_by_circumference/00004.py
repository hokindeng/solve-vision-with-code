import cv2
import numpy as np
import imageio


def solve():
    # 1. Read first frame
    img = cv2.imread('/app/first_frame.png')
    
    # 2. Find the circles
    # Background is exactly white [255, 255, 255]
    # We find black pixels for the exact bounding boxes of the circles
    black_mask = np.all(img == [0, 0, 0], axis=-1).astype(np.uint8) * 255
    contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    circles = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Extract sprite
        crop = img[y:y+h, x:x+w].copy()
        # Create mask: anything that is not white is part of the circle
        mask = np.any(crop != [255, 255, 255], axis=-1)
        circles.append({
            'start_x': x,
            'start_y': y,
            'w': w,
            'h': h,
            'sprite': crop,
            'mask': mask
        })
    
    # 3. Sort circles by circumference (equivalent to width) largest to smallest
    circles = sorted(circles, key=lambda c: c['w'], reverse=True)
    
    # 4. Calculate final positions
    gap = 15
    total_width = sum(c['w'] for c in circles) + gap * (len(circles) - 1)
    start_x = (1024 - total_width) // 2
    
    # center_y = 512.5 so that circles' true centers align exactly on the image center
    center_y = 512.5
    
    for c in circles:
        c['end_x'] = start_x
        c['end_y'] = int(center_y - c['h'] / 2)
        start_x += c['w'] + gap
        
    # 5. Generate frames
    frames = []
    num_frames = 80
    
    for i in range(num_frames):
        t = i / (num_frames - 1)
        # smoothstep easing
        progress = t * t * (3 - 2 * t)
        
        # Create a new white background
        frame = np.ones((1024, 1024, 3), dtype=np.uint8) * 255
        
        # Draw circles
        # To handle overlapping gracefully during animation, 
        # we can sort by size or just use original order.
        # Doesn't matter since they don't overlap in the end.
        for c in circles:
            curr_x = int(round(c['start_x'] + progress * (c['end_x'] - c['start_x'])))
            curr_y = int(round(c['start_y'] + progress * (c['end_y'] - c['start_y'])))
            
            # Paste sprite onto frame using mask
            h, w = c['h'], c['w']
            
            # Get ROI on frame
            roi = frame[curr_y:curr_y+h, curr_x:curr_x+w]
            
            # Apply mask
            roi[c['mask']] = c['sprite'][c['mask']]
            
        # Convert BGR to RGB for imageio
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        
    # 6. Save video
    # H.264, yuv420p, 1024x1024, 16 fps
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', pixelformat='yuv420p', macro_block_size=None)
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    
    print("Video generated successfully.")

if __name__ == '__main__':
    solve()
