import cv2
import numpy as np
import imageio
import os

def solve():
    img = cv2.imread('/app/first_frame.png')
    frames = []

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    color_dim = (0, 0, 0) # black
    color_diff = (0, 100, 255) # orange-red
    color_circle = (0, 0, 255) # red

    def draw_dim(img_canvas, x, w, val, alpha=1.0):
        overlay = img_canvas.copy()
        cv2.line(overlay, (int(x), 415), (int(x + w), 415), color_dim, 2)
        cv2.line(overlay, (int(x), 410), (int(x), 420), color_dim, 2)
        cv2.line(overlay, (int(x + w), 410), (int(x + w), 420), color_dim, 2)
        
        text = str(val)
        tw, th = cv2.getTextSize(text, font, font_scale, thickness)[0]
        cv2.putText(overlay, text, (int(x + w//2 - tw//2), 445), font, font_scale, color_dim, thickness)
        
        cv2.addWeighted(overlay, alpha, img_canvas, 1 - alpha, 0, img_canvas)

    def draw_diff(img_canvas, x, val, alpha=1.0):
        overlay = img_canvas.copy()
        text = f"+{val}"
        tw, th = cv2.getTextSize(text, font, font_scale, thickness)[0]
        cv2.putText(overlay, text, (int(x - tw//2), 445), font, font_scale, color_diff, thickness)
        cv2.addWeighted(overlay, alpha, img_canvas, 1 - alpha, 0, img_canvas)

    for i in range(60):
        frame = img.copy()

        # Step 1: Shape 1 and 2 dims
        alpha1 = min(max((i - 5) / 5.0, 0.0), 1.0)
        if alpha1 > 0:
            draw_dim(frame, 226, 31, 31, alpha1)
            draw_dim(frame, 393, 57, 57, alpha1)

        # Step 2: diff 1
        alpha2 = min(max((i - 11) / 4.0, 0.0), 1.0)
        if alpha2 > 0:
            draw_diff(frame, 331, 26, alpha2)

        # Step 3: Shape 3 dim
        alpha3 = min(max((i - 16) / 5.0, 0.0), 1.0)
        if alpha3 > 0:
            draw_dim(frame, 561, 83, 83, alpha3)

        # Step 4: diff 2
        alpha4 = min(max((i - 22) / 4.0, 0.0), 1.0)
        if alpha4 > 0:
            draw_diff(frame, 511, 26, alpha4)

        # Step 5: diff 3 and Shape 4 dim
        alpha5 = min(max((i - 27) / 5.0, 0.0), 1.0)
        if alpha5 > 0:
            draw_diff(frame, 692, 26, alpha5)
            draw_dim(frame, 728, 109, 109, alpha5)

        # Step 6: Circle correct choice
        if i >= 39:
            progress = min(max((i - 39) / 11.0, 0.0), 1.0)
            angle = int(360 * progress)
            if angle > 0:
                cx = 61 + 211 // 2
                cy = 752 + 205 // 2
                cv2.ellipse(frame, (cx, cy), (120, 120), -90, 0, angle, color_circle, 6)

        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    os.makedirs('/app/output', exist_ok=True)
    imageio.mimwrite('/app/output/video.mp4', frames, fps=16, macro_block_size=None, format='FFMPEG', codec='libx264', pixelformat='yuv420p')

if __name__ == '__main__':
    solve()
