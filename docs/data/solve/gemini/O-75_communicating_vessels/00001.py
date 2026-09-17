import cv2
import numpy as np
import os
import subprocess

def main():
    input_path = '/app/first_frame.png'
    output_dir = '/app/output'
    output_video = os.path.join(output_dir, 'video.mp4')
    frames_dir = '/tmp/frames'
    
    img_init = cv2.imread(input_path)
    if img_init is None:
        raise FileNotFoundError(f"Could not read {input_path}")
    
    fps = 16
    num_frames = 63
    time_steps = np.arange(num_frames) / fps
    
    # y-coordinates of the top of the liquid in the first frame
    y_inits = [584, 819, 689, 409, 504]
    
    # x-coordinate boundaries for the inside of the 5 tubes
    x_ranges = [(81, 171), (269, 359), (457, 547), (645, 735), (833, 923)]
    
    # Equilibrium y-coordinate for the final stable state (average level)
    y_final = 603
    
    # Colors for drawing
    color_orange = [75, 191, 255] # BGR
    color_white = [255, 255, 255] # BGR
    
    os.makedirs(frames_dir, exist_ok=True)
    
    for frame_idx, t in enumerate(time_steps):
        img = img_init.copy()
        
        for i in range(5):
            # Calculate the current liquid level based on exponential decay
            # h(t) = h_avg + (h_init - h_avg) * exp(-k * t)
            # Since y-coordinates are inverted, we can directly apply it to y
            y_curr = int(round(y_final + (y_inits[i] - y_final) * np.exp(-1.66 * t)))
            y_init = y_inits[i]
            x1, x2 = x_ranges[i]
            
            if y_curr < y_init:
                # Liquid goes up, paint orange from y_curr down to y_init
                img[y_curr:y_init, x1:x2] = color_orange
            elif y_curr > y_init:
                # Liquid goes down, paint white from y_init down to y_curr
                img[y_init:y_curr, x1:x2] = color_white
                
        cv2.imwrite(f"{frames_dir}/{frame_idx:04d}.png", img)
        
    os.makedirs(output_dir, exist_ok=True)
    
    # Compile the frames into a video using ffmpeg
    cmd = [
        'ffmpeg', '-y',
        '-framerate', str(fps),
        '-i', f'{frames_dir}/%04d.png',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        output_video
    ]
    
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

if __name__ == '__main__':
    main()
