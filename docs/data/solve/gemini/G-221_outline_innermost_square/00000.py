import cv2
import numpy as np
import subprocess
import os

def solve():
    img = cv2.imread('/app/first_frame.png')
    if img is None:
        print("Error: Could not read first_frame.png")
        return

    pixels = img.reshape(-1, 3)
    unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
    
    squares = []
    for color in unique_colors:
        mask = cv2.inRange(img, color, color)
        y_coords, x_coords = np.where(mask > 0)
        if len(x_coords) > 0:
            x_min, x_max = int(x_coords.min()), int(x_coords.max())
            y_min, y_max = int(y_coords.min()), int(y_coords.max())
            width = x_max - x_min + 1
            height = y_max - y_min + 1
            
            # Check if it's a square and not the full background
            if abs(width - height) < 5 and width < img.shape[1] - 5:
                squares.append((width, x_min, y_min, x_max, y_max, color))
                
    if not squares:
        print("No squares found!")
        return
        
    squares.sort(key=lambda x: x[0], reverse=True)
    smallest_sq = squares[-1]
    _, x_min, y_min, x_max, y_max, _ = smallest_sq
    
    # 1024x1024, 16 fps, 85 frames
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    os.makedirs('/app/output', exist_ok=True)
    out = cv2.VideoWriter('/app/output/video_temp.mp4', fourcc, 16, (1024, 1024))
    
    frames = 85
    thickness = 6
    offset = thickness // 2
    
    # Coordinates for the square outline (slightly inside so we don't bleed out)
    x1, y1 = x_min + offset, y_min + offset
    x2, y2 = x_max - offset, y_max - offset
    
    blue = (255, 0, 0) # BGR
    
    for i in range(frames):
        frame = img.copy()
        progress = i / (frames - 1)
        
        total_length = (x2 - x1) * 4
        current_length = progress * total_length
        
        if current_length > 0:
            l = min(current_length, x2 - x1)
            cv2.line(frame, (x1, y1), (x1 + int(l), y1), blue, thickness)
            current_length -= l
            
        if current_length > 0:
            l = min(current_length, y2 - y1)
            cv2.line(frame, (x2, y1), (x2, y1 + int(l)), blue, thickness)
            current_length -= l
            
        if current_length > 0:
            l = min(current_length, x2 - x1)
            cv2.line(frame, (x2, y2), (x2 - int(l), y2), blue, thickness)
            current_length -= l
            
        if current_length > 0:
            l = min(current_length, y2 - y1)
            cv2.line(frame, (x1, y2), (x1, y2 - int(l)), blue, thickness)
            current_length -= l
            
        out.write(frame)
        
    out.release()
    
    subprocess.run([
        'ffmpeg', '-y', '-i', '/app/output/video_temp.mp4',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        '/app/output/video.mp4'
    ], check=True)
    
    os.remove('/app/output/video_temp.mp4')
    print("Done")

if __name__ == '__main__':
    solve()
