import cv2
import numpy as np
import subprocess
import os

def create_video():
    img = cv2.imread('/app/first_frame.png')
    bg_color = np.array([240, 240, 240])
    diff = np.any(img != bg_color, axis=-1)
    contours, _ = cv2.findContours(diff.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    red_balls = []
    green_balls = []
    violet_balls = []

    for c in contours:
        area = cv2.contourArea(c)
        if area < 2000:
            x,y,w,h = cv2.boundingRect(c)
            cx, cy = x + w//2, y + h//2
            color = img[cy, cx]
            if color[2] > 200 and color[0] < 100: # Red
                red_balls.append((x,y,w,h))
            elif color[1] > 200 and color[0] < 200: # Green
                green_balls.append((x,y,w,h))
            elif color[2] > 200 and color[0] > 200: # Violet
                violet_balls.append((x,y,w,h))

    # Create clean background
    clean_bg = img.copy()
    for (x,y,w,h) in red_balls + green_balls + violet_balls:
        clean_bg[y:y+h, x:x+w] = bg_color

    # Extract patches
    def get_patch(box):
        x,y,w,h = box
        patch = img[y:y+h, x:x+w].copy()
        mask = np.any(patch != bg_color, axis=-1)
        return patch, mask

    red_patch, red_mask = get_patch(red_balls[0])
    green_patch, green_mask = get_patch(green_balls[0])
    violet_patch, violet_mask = get_patch(violet_balls[0])

    # Compute centers
    def get_center(boxes):
        cx = sum(x + w//2 for x,y,w,h in boxes) / len(boxes)
        cy = sum(y + h//2 for x,y,w,h in boxes) / len(boxes)
        return np.array([cx, cy])

    r_center = get_center(red_balls)
    v_center = get_center(violet_balls)
    g_center = get_center(green_balls)

    dv1 = v_center - r_center
    dv2 = g_center - v_center

    os.makedirs('/app/output', exist_ok=True)
    out = cv2.VideoWriter('/app/output/video_temp.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 16, (1024, 1024))

    def draw_balls(frame, boxes, patch, mask, offset=np.array([0,0])):
        for (x,y,w,h) in boxes:
            nx, ny = int(x + offset[0]), int(y + offset[1])
            if 0 <= nx < 1024 - w and 0 <= ny < 1024 - h:
                roi = frame[ny:ny+h, nx:nx+w]
                np.copyto(roi, patch, where=mask[..., None])

    for f in range(62):
        frame = clean_bg.copy()

        r_offset = np.array([0.0, 0.0])
        v_offset = np.array([0.0, 0.0])
        
        v_is_red = False
        g_is_red = False

        if f <= 5:
            pass
        elif f <= 20:
            r_offset = dv1 * ((f - 5) / 15.0)
        elif f <= 25:
            r_offset = dv1
        elif f <= 35:
            r_offset = dv1
            v_is_red = True
        elif f <= 50:
            r_offset = dv1 + dv2 * ((f - 35) / 15.0)
            v_offset = dv2 * ((f - 35) / 15.0)
            v_is_red = True
        elif f <= 55:
            r_offset = dv1 + dv2
            v_offset = dv2
            v_is_red = True
        else:
            r_offset = dv1 + dv2
            v_offset = dv2
            v_is_red = True
            g_is_red = True

        # Draw G
        draw_balls(frame, green_balls, red_patch if g_is_red else green_patch, red_mask if g_is_red else green_mask)
        # Draw V
        draw_balls(frame, violet_balls, red_patch if v_is_red else violet_patch, red_mask if v_is_red else violet_mask, v_offset)
        # Draw R
        draw_balls(frame, red_balls, red_patch, red_mask, r_offset)

        out.write(frame)

    out.release()
    
    # re-encode with ffmpeg to meet requirements: H.264, yuv420p
    subprocess.run(['ffmpeg', '-y', '-i', '/app/output/video_temp.mp4', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '/app/output/video.mp4'], check=True)
    os.remove('/app/output/video_temp.mp4')

if __name__ == '__main__':
    create_video()
