import cv2
import numpy as np
import os
import imageio

def create_video():
    input_path = '/app/first_frame.png'
    output_path = '/app/output/video.mp4'
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    img = cv2.imread(input_path)
    
    writer = imageio.get_writer(output_path, fps=16, codec='libx264', macro_block_size=None, format='FFMPEG', pixelformat='yuv420p')

    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 1.0
    thickness = 2

    def get_text_center(text, font, scale, thickness):
        (w, h), _ = cv2.getTextSize(text, font, scale, thickness)
        return w, h

    def put_centered_text(img, text, cx, y, color):
        w, h = get_text_center(text, font, font_scale, thickness)
        cv2.putText(img, text, (int(cx - w/2), y), font, font_scale, color, thickness, cv2.LINE_AA)

    def draw_cross(frame, cx, cy, progress):
        size = int(60 * progress)
        if size > 0:
            cv2.line(frame, (cx - size, cy - size), (cx + size, cy + size), (0, 0, 255), 5, cv2.LINE_AA)
            cv2.line(frame, (cx + size, cy - size), (cx - size, cy + size), (0, 0, 255), 5, cv2.LINE_AA)

    # Sequence centers
    c1 = 241
    c2 = 421
    c3 = 602
    c4 = 781
    y_text = 440

    for i in range(60):
        frame = img.copy()
        
        # 1. Size pattern visualization
        if i >= 5:
            put_centered_text(frame, "SMALL", c1, y_text, (0, 0, 0))
        if i >= 10:
            put_centered_text(frame, "LARGE", c2, y_text, (0, 0, 0))
        if i >= 15:
            put_centered_text(frame, "SMALL", c3, y_text, (0, 0, 0))
        if i >= 20:
            # Predict the next size in red
            put_centered_text(frame, "LARGE", c4, y_text, (0, 0, 255))
            
        # 2. Evaluate choices
        # Cross out wrong size (Choice 1)
        if i >= 25:
            p1 = min(1.0, (i - 25) / 5.0)
            draw_cross(frame, 396, 854, p1)
            
        # Cross out wrong color/shape (Choice 2 and Choice 3)
        if i >= 32:
            p2 = min(1.0, (i - 32) / 5.0)
            draw_cross(frame, 626, 854, p2)
            draw_cross(frame, 856, 854, p2)
            
        # 3. Circle the correct option (Choice 0)
        if i >= 40:
            progress = min(1.0, (i - 40) / 10.0)
            center = (166, 854)
            radius = 120
            end_angle = int(360 * progress)
            if end_angle > 0:
                cv2.ellipse(frame, center, (radius, radius), -90, 0, end_angle, (0, 0, 255), 6, cv2.LINE_AA)
                
        # Convert BGR to RGB for imageio
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(rgb_frame)

    writer.close()

if __name__ == "__main__":
    create_video()
