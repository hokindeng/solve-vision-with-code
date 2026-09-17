import cv2
import numpy as np
import imageio
import os

def main():
    grid_str = [
        "011111111111110",
        "000100000000010",
        "020111111101110",
        "010100000101000",
        "010111011101010",
        "010001010001010",
        "011101010111110",
        "000100010000010",
        "010111011111010",
        "010001010001000",
        "010111011101110",
        "010100000100010",
        "010101110101113",
        "010101010001010",
        "011111011111010"
    ]
    path = [(2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (6, 2), (6, 3), (7, 3), (8, 3), (8, 4), (8, 5), (9, 5), (10, 5), (10, 4), (10, 3), (11, 3), (12, 3), (13, 3), (14, 3), (14, 4), (14, 5), (13, 5), (12, 5), (12, 6), (12, 7), (13, 7), (14, 7), (14, 8), (14, 9), (14, 10), (14, 11), (13, 11), (12, 11), (12, 12), (12, 13), (12, 14)]

    def get_center(y, x):
        y0 = int(y * 1024 / 15)
        y1 = int((y + 1) * 1024 / 15)
        x0 = int(x * 1024 / 15)
        x1 = int((x + 1) * 1024 / 15)
        return (x0 + x1) // 2, (y0 + y1) // 2

    frames_points = []
    # Hold start
    for _ in range(5):
        frames_points.append(get_center(*path[0]))

    # Move
    for i in range(1, len(path)):
        pt_prev = get_center(*path[i-1])
        pt_curr = get_center(*path[i])
        mid = ( (pt_prev[0] + pt_curr[0])//2, (pt_prev[1] + pt_curr[1])//2 )
        frames_points.append(mid)
        frames_points.append(pt_curr)

    # Hold end
    for _ in range(12):
        frames_points.append(get_center(*path[-1]))

    os.makedirs('/app/output', exist_ok=True)
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', macro_block_size=None, pixelformat='yuv420p')

    base_frame = cv2.imread('/app/first_frame.png')
    base_frame = cv2.cvtColor(base_frame, cv2.COLOR_BGR2RGB)
    
    # Identify exactly the white pixels.
    white_mask = np.all(base_frame == [255, 255, 255], axis=-1)

    # We use a separate path canvas to draw the growing path.
    path_mask = np.zeros(base_frame.shape[:2], dtype=np.uint8)

    for i in range(len(frames_points)):
        if i > 0:
            pt_prev = frames_points[i-1]
            pt_curr = frames_points[i]
            # Draw line and circle for the sweeping brush effect
            cv2.line(path_mask, pt_prev, pt_curr, 255, 42)
            cv2.circle(path_mask, pt_curr, 21, 255, -1)
        
        # Create this frame by copying base_frame
        frame = base_frame.copy()
        
        # Apply green color where both path_mask is drawn AND pixel was originally white
        draw_condition = (path_mask == 255) & white_mask
        frame[draw_condition] = [50, 200, 50]
        
        writer.append_data(frame)

    writer.close()

if __name__ == "__main__":
    main()
