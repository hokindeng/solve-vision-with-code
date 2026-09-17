import cv2
import numpy as np
import imageio
import os

def solve():
    img = cv2.imread('/app/first_frame.png')
    
    # Identify inside points
    points_mask = np.all(img == [0, 0, 0], axis=-1).astype(np.uint8) * 255
    contours, _ = cv2.findContours(points_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    inside_overlap_pts = []
    for c in contours:
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [c], -1, 255, -1)
        
        kernel = np.ones((3,3), np.uint8)
        dilated = cv2.dilate(mask, kernel, iterations=2)
        neighbors_mask = dilated - mask
        neighbor_colors = img[neighbors_mask == 255]
        
        is_inside = True
        for color in neighbor_colors:
            if not (np.array_equal(color, [121, 166, 160]) or np.array_equal(color, [0, 0, 0])):
                is_inside = False
                break
                
        if is_inside:
            M = cv2.moments(c)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                inside_overlap_pts.append((cx, cy))

    # Sort points to make animation deterministic (e.g., top-left to bottom-right)
    inside_overlap_pts.sort(key=lambda p: (p[1], p[0]))

    frames = []
    num_frames = 37
    n_points = len(inside_overlap_pts)
    
    start_draw_frame = 6
    end_draw_frame = 25
    total_draw_frames = end_draw_frame - start_draw_frame + 1
    
    for F in range(num_frames):
        frame = img.copy()
        
        if F < start_draw_frame:
            pass
        elif F > end_draw_frame:
            for pt in inside_overlap_pts:
                cv2.circle(frame, pt, 20, (0, 0, 255), 2, lineType=cv2.LINE_8)
        else:
            progress = (F - start_draw_frame + 1) / total_draw_frames
            total_circles_to_draw = progress * n_points
            
            for i, pt in enumerate(inside_overlap_pts):
                if total_circles_to_draw >= i + 1:
                    # Fully drawn
                    cv2.circle(frame, pt, 20, (0, 0, 255), 2, lineType=cv2.LINE_8)
                elif total_circles_to_draw > i:
                    # Partially drawn
                    fraction = total_circles_to_draw - i
                    angle = int(360 * fraction)
                    if angle > 0:
                        cv2.ellipse(frame, pt, (20, 20), 0, 0, angle, (0, 0, 255), 2, lineType=cv2.LINE_8)
                        
        # Convert BGR to RGB for imageio
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    os.makedirs('/app/output', exist_ok=True)
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', format='FFMPEG', macro_block_size=None, pixelformat='yuv420p')
    for frame in frames:
        writer.append_data(frame)
    writer.close()

if __name__ == '__main__':
    solve()
