import cv2
import numpy as np
import imageio
import os

def solve():
    img_orig = cv2.imread('/app/first_frame.png')
    
    # Find background color (most frequent color)
    colors = np.unique(img_orig.reshape(-1, 3), axis=0)
    counts = [np.sum(np.all(img_orig == c, axis=-1)) for c in colors]
    bg_color = colors[np.argmax(counts)]
    
    # Create mask of all non-background pixels to find rectangles
    mask = np.any(img_orig != bg_color, axis=-1).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    rects = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 10 and h > 10:
            rects.append({'x': x, 'y': y, 'w': w, 'h': h})
            
    # Sort rectangles by y-coordinate (top to bottom)
    rects = sorted(rects, key=lambda r: r['y'])
    
    frames = []
    FPS = 16
    TOTAL_FRAMES = 48
    
    for f in range(TOTAL_FRAMES):
        img = img_orig.copy()
        
        if f < 4:
            # Pause on initial frame
            pass
        elif f < 34:
            # Phase 1: Step-by-step comparison
            f_comp = f - 4
            current_eval = f_comp // 6
            
            if current_eval < len(rects):
                r = rects[current_eval]
                x, y, w, h = r['x'], r['y'], r['w'], r['h']
                ratio = w / h
                
                # Highlight the current rectangle being evaluated
                cv2.rectangle(img, (x - 5, y - 5), (x + w + 5, y + h + 5), (255, 0, 0), 3)
                
                # Draw the ratio text
                text = f"w/h = {w}/{h} = {ratio:.2f}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8
                thickness = 2
                text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
                
                text_x = x + w + 20
                text_y = y + h // 2 + text_size[1] // 2
                
                # Text is black
                cv2.putText(img, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

        if f >= 34:
            # Phase 2: Draw red circle around the rectangle closest to a square
            # Closer to square means w/h ratio closest to 1.0
            best_idx = min(range(len(rects)), key=lambda i: max(rects[i]['w']/rects[i]['h'], rects[i]['h']/rects[i]['w']))
            best_r = rects[best_idx]
            
            cx = best_r['x'] + best_r['w'] // 2
            cy = best_r['y'] + best_r['h'] // 2
            radius = max(best_r['w'], best_r['h']) // 2 + 15
            
            # Animate the drawing of the circle over 5 frames
            progress = min(1.0, (f - 34) / 5.0)
            end_angle = int(360 * progress)
            
            if end_angle > 0:
                cv2.ellipse(img, (cx, cy), (radius, radius), 0, 0, end_angle, (0, 0, 255), 4)
                
        # Convert BGR (OpenCV) to RGB (imageio)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(img_rgb)

    # Save output video
    os.makedirs('/app/output', exist_ok=True)
    imageio.mimwrite('/app/output/video.mp4', frames, fps=FPS, codec='libx264', pixelformat='yuv420p')

if __name__ == '__main__':
    solve()
