import cv2
import numpy as np
import imageio
import os

def ease_out(t):
    return 1 - (1 - t) * (1 - t)

def main():
    img_path = '/app/first_frame.png'
    out_path = '/app/output/video.mp4'
    os.makedirs('/app/output', exist_ok=True)
    
    img = cv2.imread(img_path)
    if img is None:
        print("Error: could not read first_frame.png")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rects = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # ignore very small contours if any
        if w > 10 and h > 10:
            rects.append((x, y, w, h))
    
    # Sort by Y coordinate so we go top to bottom
    rects.sort(key=lambda r: r[1])
    
    # Find the most square rectangle
    best_idx = -1
    best_diff = float('inf')
    for i, (x, y, w, h) in enumerate(rects):
        ratio = max(w/h, h/w)
        diff = abs(1 - ratio)
        if diff < best_diff:
            best_diff = diff
            best_idx = i

    frames = []
    num_frames = 48
    
    for i in range(num_frames):
        frame = img.copy()
        
        for r_idx, (x, y, w, h) in enumerate(rects):
            start_f = 4 + r_idx * 8
            if i >= start_f:
                ratio = max(w/h, h/w)
                ratio_str = f"{w/h:.2f}" if w > h else f"{w/h:.2f}" # wait, let's just show w/h
                # Let's show ratio as w/h, it is easier to understand. 
                # Or just show the max dimension / min dimension? No, w/h is standard.
                text1 = f"W:{w} H:{h}"
                text2 = f"W/H:{w/h:.2f}"
                
                text_x = x + w + 15
                text_y = y + h//2 - 10
                
                if i >= start_f + 4:
                    cv2.putText(frame, text1, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
                    
                    # highlight the best one in green once we've reviewed all of them
                    # Review finishes at frame 4 + 2*8 + 4 = 24. Let's say at frame 26, it turns green.
                    if i >= 26 and r_idx == best_idx:
                        cv2.putText(frame, text2, (text_x, text_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,150,0), 2)
                    else:
                        cv2.putText(frame, text2, (text_x, text_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
                elif i >= start_f:
                    cv2.putText(frame, text1, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)

        # Draw red circle around the best rectangle
        if i >= 30:
            progress = (i - 30) / (47 - 30)
            progress = min(1.0, max(0.0, progress))
            
            x, y, w, h = rects[best_idx]
            cx, cy = x + w//2, y + h//2
            radius = max(w, h) // 2 + 20
            
            angle = int(360 * ease_out(progress))
            if angle > 0:
                cv2.ellipse(frame, (cx, cy), (radius, radius), 0, 0, angle, (0, 0, 255), 4)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    writer = imageio.get_writer(out_path, fps=16, codec='libx264', pixelformat='yuv420p', macro_block_size=None)
    for f in frames:
        writer.append_data(f)
    writer.close()

if __name__ == '__main__':
    main()
