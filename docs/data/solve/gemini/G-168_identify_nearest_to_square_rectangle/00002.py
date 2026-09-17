import cv2
import numpy as np
import imageio
import os

def get_rects(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    rects = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        rects.append((x, y, w, h))
    
    rects.sort(key=lambda r: r[0])
    return rects

def generate_video():
    img_base = cv2.imread('/app/first_frame.png')
    rects = get_rects('/app/first_frame.png')
    
    # Identify winner (closest to 1.0)
    best_rect = None
    best_diff = float('inf')
    for i, (x, y, w, h) in enumerate(rects):
        # We define ratio as max(w/h, h/w) to measure how close to 1:1, or simply w/h and check distance to 1
        # The prompt says "width-to-height ratio closest to 1:1"
        ratio = w / h
        diff = abs(1.0 - ratio)
        if diff < best_diff:
            best_diff = diff
            best_rect = (x, y, w, h)

    frames = []
    total_frames = 48
    
    # Timing logic
    # 0-3: hold
    # 4-9: rect 0 text fade in
    # 10-15: rect 1 text fade in
    # 16-21: rect 2 text fade in
    # 22-26: hold
    # 27-38: draw circle
    # 39-47: hold

    for f in range(total_frames):
        frame = img_base.copy()
        
        # Draw texts
        for i, (x, y, w, h) in enumerate(rects):
            ratio = w / h
            text1 = f"W:{w}, H:{h}"
            text2 = f"Ratio: {ratio:.2f}"
            
            # Determine alpha for this rect
            start_frame = 4 + i * 6
            end_frame = start_frame + 5
            
            if f < start_frame:
                alpha = 0.0
            elif f >= end_frame:
                alpha = 1.0
            else:
                alpha = (f - start_frame) / (end_frame - start_frame)
                
            if alpha > 0:
                color_val = int(255 * (1 - alpha))
                color = (color_val, color_val, color_val)
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 0.6
                thick = 2
                (w1, h1), _ = cv2.getTextSize(text1, font, scale, thick)
                (w2, h2), _ = cv2.getTextSize(text2, font, scale, thick)
                
                cx = x + w // 2
                tx1 = cx - w1 // 2
                tx2 = cx - w2 // 2
                
                cv2.putText(frame, text1, (tx1, y - 28), font, scale, color, thick, cv2.LINE_AA)
                cv2.putText(frame, text2, (tx2, y - 6), font, scale, color, thick, cv2.LINE_AA)
        
        # Draw circle on winner
        if f >= 27:
            circle_start = 27
            circle_end = 38
            if f >= circle_end:
                angle = 360
            else:
                angle = int(360 * (f - circle_start) / (circle_end - circle_start))
                
            bx, by, bw, bh = best_rect
            cx = bx + bw // 2
            cy = by + bh // 2
            radius = int(np.sqrt(bw**2 + bh**2) / 2) + 20
            
            if angle > 0:
                cv2.ellipse(frame, (cx, cy), (radius, radius), 0, -90, -90 + angle, (0, 0, 255), 4, cv2.LINE_AA)

        # Convert BGR to RGB for imageio
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    # Save to mp4
    out_path = '/app/output/video.mp4'
    writer = imageio.get_writer(out_path, fps=16, codec='libx264', format='FFMPEG', pixelformat='yuv420p')
    for frame in frames:
        writer.append_data(frame)
    writer.close()

if __name__ == '__main__':
    os.makedirs('/app/output', exist_ok=True)
    generate_video()
