import cv2
import numpy as np
import os
import subprocess

def get_state(frame):
    P0 = (343, 328); R0 = 38
    P1 = (268, 847); R1 = 65
    P2 = (769, 422); R2 = 123
    P3 = (106, 306); R3 = 201
    P4 = (910, 626); R4 = 300
    
    if frame == 0:
        return P0, R0, []
    
    eaten = []
    if frame > 17: eaten.append(0)
    if frame > 43: eaten.append(2)
    if frame > 69: eaten.append(4)
    if frame > 102: eaten.append(1)

    if frame <= 17:
        t = frame / 17.0
        cx = P0[0] + (P1[0] - P0[0]) * t
        cy = P0[1] + (P1[1] - P0[1]) * t
        r = R0
    elif frame <= 21:
        t = (frame - 17) / 4.0
        cx, cy = P1
        r = R0 + (R1 - R0) * t
    elif frame <= 43:
        t = (frame - 21) / 22.0
        cx = P1[0] + (P2[0] - P1[0]) * t
        cy = P1[1] + (P2[1] - P1[1]) * t
        r = R1
    elif frame <= 47:
        t = (frame - 43) / 4.0
        cx, cy = P2
        r = R1 + (R2 - R1) * t
    elif frame <= 69:
        t = (frame - 47) / 22.0
        cx = P2[0] + (P3[0] - P2[0]) * t
        cy = P2[1] + (P3[1] - P2[1]) * t
        r = R2
    elif frame <= 73:
        t = (frame - 69) / 4.0
        cx, cy = P3
        r = R2 + (R3 - R2) * t
    elif frame <= 102:
        t = (frame - 73) / 29.0
        cx = P3[0] + (P4[0] - P3[0]) * t
        cy = P3[1] + (P4[1] - P3[1]) * t
        r = R3
    elif frame <= 106:
        t = (frame - 102) / 4.0
        cx, cy = P4
        r = R3 + (R4 - R3) * t
    else:
        cx, cy = P4
        r = R4
        
    return (cx, cy), r, eaten

def main():
    os.makedirs('/app/output', exist_ok=True)
    first_frame = cv2.imread('/app/first_frame.png')
    
    # Pre-calculate the background image (first_frame with black ball removed)
    bg_img = first_frame.copy()
    cv2.circle(bg_img, (343, 328), 38 + 3, (255, 255, 255), -1)
    
    balls_info = {
        0: (268, 847, 27),
        1: (910, 626, 99),
        2: (769, 422, 58),
        4: (106, 306, 78)
    }
    
    frames_dir = '/app/output/frames'
    os.makedirs(frames_dir, exist_ok=True)
    
    for f in range(108):
        if f == 0:
            frame_img = first_frame.copy()
        else:
            (cx, cy), r, eaten = get_state(f)
            frame_img = bg_img.copy()
            
            # Erase eaten balls
            for b_idx in eaten:
                bx, by, br = balls_info[b_idx]
                cv2.circle(frame_img, (bx, by), br + 3, (255, 255, 255), -1)
                
            # Draw black ball
            cv2.circle(frame_img, (int(round(cx)), int(round(cy))), int(round(r)), (0, 0, 0), -1, cv2.LINE_AA)
            
        cv2.imwrite(f"{frames_dir}/frame_{f:03d}.png", frame_img)
        
    # generate video
    cmd = [
        'ffmpeg', '-y', '-framerate', '16', '-i', f"{frames_dir}/frame_%03d.png",
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '/app/output/video.mp4'
    ]
    subprocess.run(cmd, check=True)

if __name__ == '__main__':
    main()
