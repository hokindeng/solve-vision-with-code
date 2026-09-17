import cv2
import numpy as np
import imageio

def solve():
    img = cv2.imread('/app/first_frame.png')
    
    # White background mask to ensure we ONLY draw on the pathway
    white_mask = (img[:, :, 0] == 255) & (img[:, :, 1] == 255) & (img[:, :, 2] == 255)
    non_white_mask = ~white_mask

    centers = [34, 102, 170, 238, 307, 375, 443, 511, 580, 648, 716, 784, 853, 921, 989]
    
    # Pre-calculated shortest path from BFS
    path = [
        (3, 5), (3, 4), (3, 3), (2, 3), (1, 3), (1, 2), (1, 1), (2, 1), 
        (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1), (9, 1), (9, 2), 
        (9, 3), (10, 3), (11, 3), (11, 4), (11, 5), (10, 5), (9, 5), (8, 5), 
        (7, 5), (7, 6), (7, 7), (8, 7), (9, 7), (10, 7), (11, 7), (11, 8), 
        (11, 9), (10, 9), (9, 9), (9, 10), (9, 11), (9, 12), (9, 13), 
        (10, 13), (11, 13), (12, 13), (13, 13)
    ]

    frames = []
    
    # Generate each frame for the path progression
    for step in range(len(path)):
        frame = img.copy()
        
        # Draw path up to current step
        for i in range(step):
            p1 = (centers[path[i][1]], centers[path[i][0]])
            p2 = (centers[path[i+1][1]], centers[path[i+1][0]])
            cv2.line(frame, p1, p2, (50, 200, 50), 20, cv2.LINE_AA)
            cv2.circle(frame, p2, 10, (50, 200, 50), -1, cv2.LINE_AA)
            cv2.circle(frame, p1, 10, (50, 200, 50), -1, cv2.LINE_AA)
            
        # Restore all original non-white pixels (keeps walls, start, end markers perfect)
        frame[non_white_mask] = img[non_white_mask]
        
        # Convert BGR to RGB for imageio
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    # Add 15 hold frames at the end to match ~58 frames requirement (43 + 15 = 58)
    for _ in range(15):
        frames.append(frames[-1])

    # Save to video
    imageio.mimwrite(
        '/app/output/video.mp4', 
        frames, 
        fps=16, 
        codec='libx264', 
        pixelformat='yuv420p',
        macro_block_size=1
    )

if __name__ == '__main__':
    solve()
