import cv2
import numpy as np
import os
import subprocess

def create_video():
    # Ensure output directory exists
    os.makedirs('/app/output', exist_ok=True)
    
    # Read the first frame
    first_frame = cv2.imread('/app/first_frame.png')
    
    # Create a clean background by removing the agent and filling with the green cell color
    clean_bg = first_frame.copy()
    
    # The agent is perfectly aliased with orange pixels
    # BGR format: Orange is roughly [0, 165, 255]
    orange_mask = cv2.inRange(clean_bg, np.array([0, 160, 250]), np.array([5, 170, 255]))
    
    # Fill the original agent's position with the green cell color
    clean_bg[orange_mask > 0] = [0, 255, 0]
    
    # Extract the agent's pixels to paste later
    # The agent bounding box is exactly X:[935, 1003], Y:[323, 391] (69x69)
    agent_patch = first_frame[323:392, 935:1004].copy()
    agent_mask = orange_mask[323:392, 935:1004]
    
    # Define the path using Grid coordinates (Col, Row)
    # Start: (9,3) -> Green cell
    # Y1: (3,4)    -> Yellow number 1
    # Y2: (6,8)    -> Yellow number 2
    # Y3: (0,0)    -> Yellow number 3
    # End: (3,3)   -> Red cell
    #
    # To optimize path and prevent retracing visually:
    # 1. (9,3) -> (9,4) -> (3,4) [len 1 + 6 = 7]
    # 2. (3,4) -> (3,8) -> (6,8) [len 4 + 3 = 7]
    # 3. (6,8) -> (6,0) -> (0,0) [len 8 + 6 = 14]
    # 4. (0,0) -> (0,3) -> (3,3) [len 3 + 3 = 6]
    segments = [
        ((9,3), (9,4), 1),
        ((9,4), (3,4), 6),
        ((3,4), (3,8), 4),
        ((3,8), (6,8), 3),
        ((6,8), (6,0), 8),
        ((6,0), (0,0), 6),
        ((0,0), (0,3), 3),
        ((0,3), (3,3), 3)
    ]
    
    # Helper to interpolate position along the path based on linear distance
    def get_pos(d):
        for start, end, length in segments:
            if d <= length + 1e-9:
                ratio = d / length
                c = start[0] + (end[0] - start[0]) * ratio
                r = start[1] + (end[1] - start[1]) * ratio
                return c, r
            d -= length
        return segments[-1][1]
    
    temp_mp4 = '/tmp/temp_video.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_mp4, fourcc, 16.0, (1024, 1024))
    
    total_frames = 122
    max_d = sum(seg[2] for seg in segments)
    
    for f in range(total_frames):
        # Progress evenly over 122 frames (0 to 121)
        d = f * max_d / (total_frames - 1)
        c, r = get_pos(d)
        
        # Convert Grid coordinates to pixel coordinates
        # Grid cell (C,R) top-left of the agent is at X = C * 102 + 17, Y = R * 102 + 17
        x = int(round(c * 102 + 17))
        y = int(round(r * 102 + 17))
        
        frame = clean_bg.copy()
        
        # Paste the agent onto the frame using its original mask
        h, w = agent_patch.shape[:2]
        roi = frame[y:y+h, x:x+w]
        np.copyto(roi, agent_patch, where=(agent_mask > 0)[:, :, None])
        
        out.write(frame)
        
    out.release()
    
    # Encode with ffmpeg to required specs: H.264, yuv420p
    cmd = [
        'ffmpeg', '-y', '-i', temp_mp4,
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        '/app/output/video.mp4'
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Clean up temp file
    if os.path.exists(temp_mp4):
        os.remove(temp_mp4)

if __name__ == '__main__':
    create_video()
