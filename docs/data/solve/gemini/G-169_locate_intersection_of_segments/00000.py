import cv2
import numpy as np
import imageio
import os

def get_intersection(img):
    c1 = np.array([173,  68, 142])
    c2 = np.array([231,  92, 108])

    mask1 = cv2.inRange(img, c1, c1)
    mask2 = cv2.inRange(img, c2, c2)

    y1, x1 = np.where(mask1 > 0)
    y2, x2 = np.where(mask2 > 0)

    pts1 = np.column_stack((x1, y1)).astype(np.float32)
    pts2 = np.column_stack((x2, y2)).astype(np.float32)

    line1 = cv2.fitLine(pts1, cv2.DIST_L2, 0, 0.01, 0.01)
    line2 = cv2.fitLine(pts2, cv2.DIST_L2, 0, 0.01, 0.01)

    vx1, vy1, x01, y01 = line1.ravel()
    vx2, vy2, x02, y02 = line2.ravel()

    A = np.array([[vx1, -vx2], [vy1, -vy2]])
    B = np.array([x02 - x01, y02 - y01])
    t = np.linalg.solve(A, B)
    t1, t2 = t
    inter_x = x01 + t1 * vx1
    inter_y = y01 + t1 * vy1
    return int(round(inter_x)), int(round(inter_y))

def main():
    img = cv2.imread('/app/first_frame.png')
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cx, cy = get_intersection(img)
    
    frames = []
    num_frames = 30
    red = (255, 0, 0) # RGB
    
    for i in range(num_frames):
        frame = img_rgb.copy()
        
        # Step 1: Locate (dot)
        if i >= 4:
            r_dot = int(np.interp(i, [4, 9], [0, 5]))
            if r_dot > 0:
                cv2.circle(frame, (cx, cy), r_dot, red, -1, cv2.LINE_AA)
                
        # Step 2: Draw circle around it
        if i >= 12:
            angle = np.interp(i, [12, 26], [0, 360])
            if angle > 0:
                cv2.ellipse(frame, (cx, cy), (25, 25), 0, 0, angle, red, 3, cv2.LINE_AA)
                
        frames.append(frame)
        
    os.makedirs('/app/output', exist_ok=True)
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', format='FFMPEG', macro_block_size=None, pixelformat='yuv420p')
    for frame in frames:
        writer.append_data(frame)
    writer.close()

if __name__ == '__main__':
    main()
