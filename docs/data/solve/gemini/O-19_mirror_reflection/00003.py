import cv2
import numpy as np
import os
import subprocess

def main():
    img = cv2.imread('/app/first_frame.png')
    
    # Create background image with annotation removed
    bg_img = img.copy()
    annotation_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    
    # Bounding box for annotation (found empirically)
    for y in range(370, 410):
        for x in range(410, 590):
            b, g, r = img[y, x]
            # If dark pixel (part of text/arc)
            if (b < 200 or g < 200 or r < 200) and not (b > 200 and g < 100 and r < 100):
                # Protect the normal line (gray-ish near x=453)
                if 452 <= x <= 454 and abs(int(b)-150) < 30 and abs(int(g)-150) < 30 and abs(int(r)-150) < 30:
                    continue
                annotation_mask[y, x] = 255

    # Dilate slightly to remove antialiasing of the black text/arc
    kernel = np.ones((3,3), np.uint8)
    annotation_mask = cv2.dilate(annotation_mask, kernel, iterations=1)
    
    # Ensure blue incident ray and normal line are strictly protected
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            b, g, r = img[y, x]
            # Protect pure blue
            if b > 200 and g < 100 and r < 100:
                annotation_mask[y, x] = 0
            # Protect normal line
            if 452 <= x <= 454 and abs(int(b)-150) < 30 and abs(int(g)-150) < 30 and abs(int(r)-150) < 30:
                annotation_mask[y, x] = 0

    bg_img[annotation_mask > 0] = (255, 255, 255)

    # Calculate reflected ray
    # Incident ray hits mirror at x=453, y=416
    start_pt = (453, 416)
    
    # Slope of incident ray is 0.6903. Reflected ray slope is -0.6903.
    # We want it to reach the right edge of the image (x=1023)
    m = -0.6903
    end_x = 1023
    end_y = int(start_pt[1] + m * (end_x - start_pt[0]))
    end_pt = (end_x, end_y)

    os.makedirs('/app/output', exist_ok=True)
    num_frames = 35
    
    for i in range(num_frames):
        # First frame is EXACTLY the input image
        if i == 0:
            frame = img.copy()
        else:
            frame = bg_img.copy()
            progress = i / (num_frames - 1)
            
            cur_x = int(start_pt[0] + progress * (end_x - start_pt[0]))
            cur_y = int(start_pt[1] + progress * (end_y - start_pt[1]))
            
            # Draw the reflected ray on an overlay for alpha blending
            overlay = frame.copy()
            cv2.line(overlay, start_pt, (cur_x, cur_y), (255, 0, 0), 5, cv2.LINE_AA)
            
            # Draw arrowhead at the tip
            angle = np.arctan2(end_y - start_pt[1], end_x - start_pt[0])
            arrow_length = 15
            arrow_angle = 0.5
            pt1 = (
                int(cur_x - arrow_length * np.cos(angle - arrow_angle)),
                int(cur_y - arrow_length * np.sin(angle - arrow_angle))
            )
            pt2 = (
                int(cur_x - arrow_length * np.cos(angle + arrow_angle)),
                int(cur_y - arrow_length * np.sin(angle + arrow_angle))
            )
            cv2.line(overlay, (cur_x, cur_y), pt1, (255, 0, 0), 3, cv2.LINE_AA)
            cv2.line(overlay, (cur_x, cur_y), pt2, (255, 0, 0), 3, cv2.LINE_AA)
            
            # Reflectivity = 0.97
            cv2.addWeighted(overlay, 0.97, frame, 0.03, 0, frame)

        cv2.imwrite(f'/app/output/frame_{i:04d}.png', frame)

    cmd = [
        'ffmpeg', '-y', '-framerate', '16', '-i', '/app/output/frame_%04d.png',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '/app/output/video.mp4'
    ]
    subprocess.run(cmd, check=True)

if __name__ == '__main__':
    main()
