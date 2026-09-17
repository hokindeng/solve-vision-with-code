import cv2
import numpy as np
import imageio

def solve():
    first_frame = cv2.imread('/app/first_frame.png')
    
    boxes = [
        (218, 137, 166, 109), # 15
        (533, 216, 212, 111), # 83_top
        (449, 300, 215, 109), # 87
        (679, 325, 176, 107), # 44
        (448, 492, 171, 111), # 31
        (597, 668, 170, 111)  # 83_bot
    ]
    
    # 87 is the largest
    target_box = boxes[2]
    center = (target_box[0] + target_box[2]//2, target_box[1] + target_box[3]//2)
    axes = (target_box[2]//2 + 20, target_box[3]//2 + 20)
    
    frames = []
    
    for f in range(80):
        frame = first_frame.copy()
        
        if f < 48:
            box_idx = f // 8
            if box_idx < len(boxes):
                x, y, w, h = boxes[box_idx]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 200, 255), 4)
        else:
            # draw the red circle step by step
            progress = min(1.0, (f - 48) / (69 - 48))
            end_angle = int(-90 + progress * 360)
            cv2.ellipse(frame, center, axes, 0, -90, end_angle, (0, 0, 255), 6)
            
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
    imageio.mimwrite('/app/output/video.mp4', frames, fps=16, codec='libx264', pixelformat='yuv420p')

if __name__ == '__main__':
    solve()
