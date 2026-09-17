import cv2
import numpy as np
import os
import subprocess
import shutil

def find_shortest_bar(img):
    # Plot area
    x_start, y_start = 143, 153
    w_plot, h_plot = 729, 729
    plot_area = img[y_start:y_start+h_plot, x_start:x_start+w_plot]

    # Convert to HSV to find colored bars (filter out white/gray/black background/axes)
    hsv = cv2.cvtColor(plot_area, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 50, 50), (180, 255, 255))
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bars = []
    for cnt in contours:
        cx, cy, cw, ch = cv2.boundingRect(cnt)
        if cw > 10 and ch > 10:
            bars.append((cx + x_start, cy + y_start, cw, ch))
            
    if not bars:
        return None
        
    # The bar with minimum height is the minimum value bar
    shortest_bar = min(bars, key=lambda b: b[3])
    return shortest_bar

def draw_partial_rect(img, rect, fraction, color=(0, 0, 255), thickness=4):
    x, y, w, h = rect
    
    # 4 corners
    p0 = (x, y)
    p1 = (x + w, y)
    p2 = (x + w, y + h)
    p3 = (x, y + h)
    
    perimeter = 2 * (w + h)
    draw_len = int(perimeter * fraction)
    
    segments = [
        (p0, p1, w),
        (p1, p2, h),
        (p2, p3, w),
        (p3, p0, h)
    ]
    
    drawn = 0
    for start_pt, end_pt, seg_len in segments:
        if drawn >= draw_len:
            break
            
        if drawn + seg_len <= draw_len:
            cv2.line(img, start_pt, end_pt, color, thickness)
            drawn += seg_len
        else:
            rem = draw_len - drawn
            ratio = rem / seg_len
            part_pt = (
                int(start_pt[0] + (end_pt[0] - start_pt[0]) * ratio),
                int(start_pt[1] + (end_pt[1] - start_pt[1]) * ratio)
            )
            cv2.line(img, start_pt, part_pt, color, thickness)
            drawn += rem

def main():
    img_path = '/app/first_frame.png'
    out_dir = '/app/output'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'video.mp4')
    
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read {img_path}")
        
    shortest_bar = find_shortest_bar(img)
    if not shortest_bar:
        print("No bars found!")
        return
        
    # Add padding to border so it's drawn *around* the bar
    pad = 4
    bx, by, bw, bh = shortest_bar
    rect = (bx - pad, by - pad, bw + 2*pad, bh + 2*pad)
    
    frames = 48
    fps = 16
    
    temp_dir = '/app/temp_frames'
    os.makedirs(temp_dir, exist_ok=True)
    
    for i in range(frames):
        frame = img.copy()
        fraction = i / (frames - 1.0)
        draw_partial_rect(frame, rect, fraction, color=(0, 0, 255), thickness=4)
        cv2.imwrite(f'{temp_dir}/frame_{i:04d}.png', frame)
        
    cmd = [
        'ffmpeg', '-y',
        '-framerate', str(fps),
        '-i', f'{temp_dir}/frame_%04d.png',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        out_path
    ]
    
    subprocess.run(cmd, check=True)
    print("Video generated!")
    
    # Cleanup temp frames
    shutil.rmtree(temp_dir)
    
if __name__ == '__main__':
    main()
