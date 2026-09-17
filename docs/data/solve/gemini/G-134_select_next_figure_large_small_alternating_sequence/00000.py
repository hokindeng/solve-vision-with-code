import cv2
import numpy as np
import os
import imageio

def get_text_size(text, font_scale=0.8, thickness=2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    return cv2.getTextSize(text, font, font_scale, thickness)[0]

def put_text(img, text, center_x, y, alpha=1.0, color=(0,0,0)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.8
    thickness = 2
    size = get_text_size(text, scale, thickness)
    x = int(center_x - size[0] // 2)
    y = int(y)
    
    if alpha < 1.0:
        overlay = img.copy()
        cv2.putText(overlay, text, (x, y), font, scale, (255, 255, 255), thickness + 2)
        cv2.putText(overlay, text, (x, y), font, scale, color, thickness)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    else:
        cv2.putText(img, text, (x, y), font, scale, (255, 255, 255), thickness + 2)
        cv2.putText(img, text, (x, y), font, scale, color, thickness)

def draw_dashed_rect(img, pt1, pt2, color, thickness=2, dash_length=10, alpha=1.0):
    if alpha < 1.0:
        overlay = img.copy()
    else:
        overlay = img
        
    x1, y1 = int(pt1[0]), int(pt1[1])
    x2, y2 = int(pt2[0]), int(pt2[1])
    
    # top edge
    for x in range(x1, x2, dash_length * 2):
        cv2.line(overlay, (x, y1), (min(x + dash_length, x2), y1), color, thickness)
    # bottom edge
    for x in range(x1, x2, dash_length * 2):
        cv2.line(overlay, (x, y2), (min(x + dash_length, x2), y2), color, thickness)
    # left edge
    for y in range(y1, y2, dash_length * 2):
        cv2.line(overlay, (x1, y), (x1, min(y + dash_length, y2)), color, thickness)
    # right edge
    for y in range(y1, y2, dash_length * 2):
        cv2.line(overlay, (x2, y), (x2, min(y + dash_length, y2)), color, thickness)
        
    if alpha < 1.0:
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

def generate_video():
    input_img_path = '/app/first_frame.png'
    output_video_path = '/app/output/video.mp4'
    
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    base_img = cv2.imread(input_img_path)
    
    if base_img is None:
        print(f"Error: could not read {input_img_path}")
        return
        
    h, w, _ = base_img.shape
    fps = 16
    total_frames = 60
    
    writer = imageio.get_writer(output_video_path, fps=fps, codec='libx264', pixelformat='yuv420p')
    
    for frame in range(total_frames):
        img = base_img.copy()
        
        # Event 1: SMALL
        a1 = np.clip((frame - 0) / 5.0, 0.0, 1.0)
        if a1 > 0:
            put_text(img, "SMALL", 241, 410, a1)
            
        # Event 2: LARGE
        a2 = np.clip((frame - 6) / 5.0, 0.0, 1.0)
        if a2 > 0:
            put_text(img, "LARGE", 421, 410, a2)
            
        # Event 3: SMALL
        a3 = np.clip((frame - 12) / 5.0, 0.0, 1.0)
        if a3 > 0:
            put_text(img, "SMALL", 602, 410, a3)
            
        # Event 4: LARGE?
        a4 = np.clip((frame - 18) / 7.0, 0.0, 1.0)
        if a4 > 0:
            put_text(img, "LARGE ?", 782, 410, a4, color=(0, 0, 255))
            draw_dashed_rect(img, (728, 283), (728+109, 283+109), (0, 0, 255), thickness=3, alpha=a4)
            
        # Event 5: Circle choice 3
        if frame >= 30:
            progress = np.clip((frame - 30) / 25.0, 0.0, 1.0)
            end_angle = int(360 * progress)
            if end_angle > 0:
                cv2.ellipse(img, (856, 854), (120, 120), 0, 0, end_angle, (0, 0, 255), 6)
                
        # Convert BGR to RGB for imageio
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        writer.append_data(img_rgb)
        
    writer.close()
    print(f"Video saved to {output_video_path}")

if __name__ == '__main__':
    generate_video()
