import cv2
import numpy as np
import imageio

def get_clean_background(img, ball_color):
    bg = img.copy()
    mask = cv2.inRange(bg, np.array(ball_color), np.array(ball_color))
    bg[mask > 0] = [255, 255, 255]
    return bg

def main():
    img = cv2.imread('/app/first_frame.png')
    
    # Ball properties
    ball_color = [30, 105, 210]
    radius = 30
    
    # Get clean background
    bg = get_clean_background(img, ball_color)
    
    # Simulation properties
    x_start = 765
    y_start = 280
    v_x_dir = 160
    v_y_dir = -6.5
    
    # Boundaries for the center of the ball
    x_min = 63 + radius
    x_max = 961 - radius
    y_min = 63 + radius
    y_max = 961 - radius
    
    # We know it hits x_max first, then bounces back and forth.
    # It stops after the 5th bounce (which is on the right wall, x_max).
    dist_to_first = x_max - x_start
    dist_bounce = x_max - x_min
    total_x_dist = dist_to_first + 4 * dist_bounce
    
    num_frames = 80
    fps = 16
    
    # Calculate positions
    frames = []
    
    for i in range(num_frames):
        # Progress from 0 to 1
        t = i / (num_frames - 1)
        current_x_dist = total_x_dist * t
        
        # Calculate y
        y = y_start + current_x_dist * (v_y_dir / v_x_dir)
        
        # Calculate x with folding
        if current_x_dist <= dist_to_first:
            x = x_start + current_x_dist
        else:
            rem = current_x_dist - dist_to_first
            bounces = int(rem // dist_bounce)
            rem_dist = rem % dist_bounce
            
            if bounces % 2 == 0:
                # Moving left
                x = x_max - rem_dist
            else:
                # Moving right
                x = x_min + rem_dist
                
        # Draw frame
        frame = bg.copy()
        cv2.circle(frame, (int(round(x)), int(round(y))), radius, ball_color, -1)
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        
    # First frame should be the exact original image
    frames[0] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Write video
    imageio.mimwrite('/app/output/video.mp4', frames, fps=fps, macro_block_size=1, format='FFMPEG', codec='libx264', pixelformat='yuv420p')

if __name__ == '__main__':
    main()
