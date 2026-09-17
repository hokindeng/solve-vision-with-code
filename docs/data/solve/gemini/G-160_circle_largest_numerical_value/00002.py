import cv2
import numpy as np
import imageio

def draw_arc(image, center, radius, start_angle, end_angle, color, thickness):
    """
    Draw an arc (part of a circle).
    """
    axes = (radius, radius)
    angle = 0
    center = (int(center[0]), int(center[1]))
    cv2.ellipse(image, center, axes, angle, start_angle, end_angle, color, thickness, lineType=cv2.LINE_AA)

def main():
    img_path = '/app/first_frame.png'
    base_img = cv2.imread(img_path)
    
    # Check resolution
    h, w, c = base_img.shape
    if (w, h) != (1024, 1024):
        base_img = cv2.resize(base_img, (1024, 1024))
        
    # Bounding boxes for each number
    groups = [
        {"val": 21, "box": (739, 888, 217, 306)},
        {"val": 15, "box": (337, 481, 244, 333)},
        {"val": 90, "box": (208, 363, 366, 457)},
        {"val": 28, "box": (550, 700, 387, 478)},
        {"val": 57, "box": (751, 899, 444, 533)},
        {"val": 33, "box": (451, 600, 521, 612)}
    ]
    
    # Sequence for "compare all numbers": left-to-right, top-to-bottom
    seq = [1, 0, 2, 3, 4, 5]
    
    frames = []
    
    total_frames = 80
    fps = 16
    
    init_pause = 5
    scan_frames_per_num = 5
    scan_end_frame = init_pause + len(seq) * scan_frames_per_num  # 5 + 30 = 35
    
    circle_start_frame = 40
    circle_duration = total_frames - circle_start_frame  # 80 - 40 = 40
    
    for f in range(total_frames):
        frame = base_img.copy()
        
        if f < init_pause:
            pass # Keep unmodified
        elif f < scan_end_frame:
            # Highlight current number to represent the comparison step
            idx = (f - init_pause) // scan_frames_per_num
            g = groups[seq[idx]]
            x_min, x_max, y_min, y_max = g["box"]
            
            # Draw blue underline (BGR: 255, 0, 0)
            cv2.line(frame, (x_min, y_max + 10), (x_max, y_max + 10), (255, 0, 0), 4, lineType=cv2.LINE_AA)
            
        elif f < circle_start_frame:
            pass # Pause before circling, keeping the canvas clean
            
        else:
            # Circle phase around 90 (group index 2)
            # Hardcoded center and radius carefully calculated to encapsulate 90 and not touch 15
            cx = 285
            cy = 415
            radius = 93
            
            progress = (f - circle_start_frame) / (circle_duration - 1)
            end_angle = int(360 * progress)
            
            if end_angle > 0:
                # Draw red circle (BGR: 0, 0, 255)
                draw_arc(frame, (cx, cy), radius, 0, end_angle, (0, 0, 255), 6)
                
        # Convert BGR to RGB for imageio
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        
    # Write video
    imageio.mimwrite('/app/output/video.mp4', frames, fps=fps, macro_block_size=None, codec='libx264', pixelformat='yuv420p')

if __name__ == '__main__':
    main()
