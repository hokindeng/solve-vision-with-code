import cv2
import numpy as np
import imageio
import os

def main():
    img = cv2.imread('/app/first_frame.png')
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 1. Analyze the image dynamically
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    lines = []
    points = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > img.shape[1] * 0.8 and h < 20: # horizontal lines
            lines.append((x, y, w, h))
        elif w < 100 and h < 100:
            points.append((x, y, w, h))

    lines.sort(key=lambda item: item[1])
    if len(lines) < 2:
        print("Could not find two horizontal lines.")
        return
        
    upper_line = lines[0]
    lower_line = lines[-1]

    y1_center = upper_line[1] + upper_line[3] / 2.0
    y2_center = lower_line[1] + lower_line[3] / 2.0
    mid_y = (y1_center + y2_center) / 2.0

    closest_point = None
    min_diff = float('inf')
    for p in points:
        px, py, pw, ph = p
        pmid_y = py + ph / 2.0
        diff = abs(pmid_y - mid_y)
        if diff < min_diff:
            min_diff = diff
            closest_point = p

    if closest_point is None:
        print("Could not find midpoint.")
        return

    px, py, pw, ph = closest_point
    p_center_x = px + pw / 2.0

    # Line extends from bottom of upper line to top of lower line
    line_y_start = upper_line[1] + upper_line[3]
    line_y_end = lower_line[1]

    # Line width same as upper line height
    thickness = upper_line[3]
    line_x_start = int(p_center_x - thickness / 2.0 + 0.5)
    line_x_end = line_x_start + thickness
    
    # 2. Animate drawing the line over 50 frames
    frames = []
    num_frames = 50
    bg_color_rgb = img_rgb[0, 0]
    
    for f in range(num_frames):
        cur = img_rgb.copy()
        
        # Calculate current bottom Y of the drawn line
        y_current = line_y_start + int(round((line_y_end - line_y_start) * (f / float(num_frames - 1))))
        
        for y in range(line_y_start, y_current):
            for x in range(line_x_start, line_x_end):
                if (cur[y, x] == bg_color_rgb).all():
                    cur[y, x] = [255, 0, 0] # Red in RGB
                    
        frames.append(cur)
        
    # 3. Save the video
    os.makedirs('/app/output', exist_ok=True)
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', pixelformat='yuv420p', macro_block_size=None)
    for frame in frames:
        writer.append_data(frame)
    writer.close()

if __name__ == '__main__':
    main()
