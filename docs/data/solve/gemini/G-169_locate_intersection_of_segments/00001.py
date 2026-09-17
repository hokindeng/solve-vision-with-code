import cv2
import numpy as np
import imageio
import os

def find_intersection_by_color(img):
    pixels = img.reshape(-1, 3)
    unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
    
    sorted_indices = np.argsort(-counts)
    unique_colors = unique_colors[sorted_indices]
    counts = counts[sorted_indices]
    
    line_colors = []
    for c, count in zip(unique_colors[1:], counts[1:]):
        if count > 100:
            line_colors.append(c)
            
    if len(line_colors) >= 2:
        mask1 = cv2.inRange(img, line_colors[0], line_colors[0])
        mask2 = cv2.inRange(img, line_colors[1], line_colors[1])
        
        y1, x1 = np.where(mask1 > 0)
        y2, x2 = np.where(mask2 > 0)
        
        if len(x1) == 0 or len(x2) == 0:
            return None
            
        vx, vy, x, y = cv2.fitLine(np.column_stack((x1, y1)), cv2.DIST_L2, 0, 0.01, 0.01)
        vx2, vy2, x2_, y2_ = cv2.fitLine(np.column_stack((x2, y2)), cv2.DIST_L2, 0, 0.01, 0.01)
        
        A = np.array([[vx[0], -vx2[0]], [vy[0], -vy2[0]]])
        b = np.array([x2_[0] - x[0], y2_[0] - y[0]])
        try:
            t = np.linalg.solve(A, b)
            int_x = x[0] + t[0] * vx[0]
            int_y = y[0] + t[0] * vy[0]
            return int(round(int_x)), int(round(int_y))
        except np.linalg.LinAlgError:
            pass
    return None

def find_intersection_fallback(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 100)

    if lines is not None:
        lines = lines[:, 0, :]
        
        group1 = []
        group2 = []
        
        t0 = lines[0][1]
        for r, t in lines:
            diff = abs(t - t0)
            diff = min(diff, np.pi - diff)
            
            if diff < 0.5:
                group1.append((r, t))
            else:
                group2.append((r, t))
                
        group1 = np.array(group1)
        group2 = np.array(group2)
        
        def average_line(group):
            if len(group) == 0:
                return 0, 0
            rhos = np.copy(group[:, 0])
            thetas = np.copy(group[:, 1])
            
            if np.max(thetas) - np.min(thetas) > np.pi / 2:
                for i in range(len(thetas)):
                    if thetas[i] > np.pi / 2:
                        thetas[i] -= np.pi
                        rhos[i] = -rhos[i]
                        
            return np.mean(rhos), np.mean(thetas)

        if len(group1) > 0 and len(group2) > 0:
            r1, t1 = average_line(group1)
            r2, t2 = average_line(group2)
            
            A = np.array([
                [np.cos(t1), np.sin(t1)],
                [np.cos(t2), np.sin(t2)]
            ])
            b = np.array([r1, r2])
            try:
                int_pt = np.linalg.solve(A, b)
                return int(round(int_pt[0])), int(round(int_pt[1]))
            except np.linalg.LinAlgError:
                pass
    return None

def solve():
    img = cv2.imread('/app/first_frame.png')
    
    intersection = find_intersection_by_color(img)
    if intersection is None:
        intersection = find_intersection_fallback(img)
        
    if intersection is None:
        # Fallback to the center of the image if somehow everything fails
        intersection = (img.shape[1] // 2, img.shape[0] // 2)

    os.makedirs('/app/output', exist_ok=True)
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', pixelformat='yuv420p', macro_block_size=None)
    
    num_frames = 30
    
    for i in range(num_frames):
        frame = img.copy()
        
        if i > 0:
            # i goes from 1 to num_frames - 1
            # proportion goes from 1/29 to 1.0
            proportion = i / (num_frames - 1)
            end_angle = proportion * 360
            cv2.ellipse(frame, intersection, (40, 40), 0, 0, end_angle, (0, 0, 255), 4, cv2.LINE_AA)
            
        # Convert BGR to RGB for imageio
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_rgb)
        
    writer.close()

if __name__ == '__main__':
    solve()
