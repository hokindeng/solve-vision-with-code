import cv2
import numpy as np
import os
import subprocess

def main():
    img_path = '/app/first_frame.png'
    img = cv2.imread(img_path)
    
    # Find the wave pixels
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = gray == 0
    y_coords = {}
    
    for x in range(mask.shape[1]):
        y_idx = np.where(mask[:, x])[0]
        if len(y_idx) > 0:
            y_coords[x] = np.mean(y_idx)
            
    x_list = sorted(y_coords.keys())
    peaks = []
    
    # Find strict local minima in y (which corresponds to local maxima in wave)
    for i in range(1, len(x_list) - 1):
        curr_x = x_list[i]
        prev_x = x_list[i-1]
        next_x = x_list[i+1]
        
        # Check if immediate neighbors exist (contiguous in x)
        if prev_x == curr_x - 1 and next_x == curr_x + 1:
            if y_coords[curr_x] < y_coords[prev_x] and y_coords[curr_x] < y_coords[next_x]:
                peaks.append((curr_x, y_coords[curr_x]))
                
    # Prepare frames
    os.makedirs('/app/output', exist_ok=True)
    
    actions = []
    for p in peaks:
        actions.append((p, 'dot'))
        actions.append((p, 'outline'))
        
    frames = []
    num_frames = 10
    
    for i in range(num_frames):
        # We have len(actions) = 6. 
        # i goes from 0 to 9.
        # min(i, 6) gives 0 actions for frame 0, up to 6 actions for frames 6-9.
        num_a = min(i, len(actions))
        frame = img.copy()
        
        for a in range(num_a):
            pt, a_type = actions[a]
            pt_int = (int(pt[0]), int(round(pt[1])))
            
            if a_type == 'dot':
                cv2.circle(frame, pt_int, 5, (0, 0, 255), -1, lineType=cv2.LINE_AA)
            elif a_type == 'outline':
                cv2.circle(frame, pt_int, 20, (0, 0, 255), 3, lineType=cv2.LINE_AA)
                
        frames.append(frame)
        cv2.imwrite(f'/app/output/frame_{i:02d}.png', frame)
        
    # Generate video
    cmd = [
        'ffmpeg', '-y', '-framerate', '16', '-i', '/app/output/frame_%02d.png',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '/app/output/video.mp4'
    ]
    subprocess.run(cmd, check=True)
    
    # Cleanup frame images
    for i in range(num_frames):
        try:
            os.remove(f'/app/output/frame_{i:02d}.png')
        except OSError:
            pass

if __name__ == '__main__':
    main()
