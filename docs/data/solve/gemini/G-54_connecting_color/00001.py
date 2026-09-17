import cv2
import numpy as np
import imageio
import os

def main():
    os.makedirs('/app/output', exist_ok=True)
    
    img = cv2.imread('/app/first_frame.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    bg_color = img[0, 0]
    mask = np.any(img != bg_color, axis=-1).astype(np.uint8) * 255
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    centers = []
    colors = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area < 1000:
            continue
        cx, cy = centroids[i]
        coords = np.where(labels[y:y+h, x:x+w] == i)
        color = img[y+coords[0][0], x+coords[1][0]]
        centers.append((int(cx), int(cy)))
        colors.append(tuple(int(c) for c in color))
    
    color_dict = {}
    for center, color in zip(centers, colors):
        if color not in color_dict:
            color_dict[color] = []
        color_dict[color].append(center)
    
    sorted_colors = sorted(color_dict.keys(), key=lambda c: sum(p[1] for p in color_dict[c])/len(color_dict[c]))
    
    def get_bezier_curve(p1, p2, offset=-80):
        p1 = np.array(p1)
        p2 = np.array(p2)
        mid = (p1 + p2) / 2
        
        dir = p2 - p1
        length = np.linalg.norm(dir)
        normal = np.array([-dir[1], dir[0]]) / length
        
        cp = mid + normal * offset
        
        t = np.linspace(0, 1, 100)
        curve = (1-t)**2 * p1[:, None] + 2*(1-t)*t * cp[:, None] + t**2 * p2[:, None]
        return curve.T.astype(np.int32)
    
    frames = []
    total_frames = 48
    fps = 16
    
    frames.append(img.copy())
    remaining_frames = total_frames - 1
    frames_per_curve = remaining_frames // len(sorted_colors)
    
    current_img = img.copy()
    
    for i, color in enumerate(sorted_colors):
        pts = sorted(color_dict[color], key=lambda p: p[0])
        p1, p2 = pts[0], pts[1]
        
        curve_pts = get_bezier_curve(p1, p2, offset=-80)
        
        num_f = frames_per_curve
        if i == len(sorted_colors) - 1:
            num_f = remaining_frames - i * frames_per_curve
            
        for f in range(1, num_f + 1):
            fraction = f / num_f
            num_pts_to_draw = max(2, int(fraction * len(curve_pts)))
            pts_to_draw = curve_pts[:num_pts_to_draw]
            
            frame = current_img.copy()
            cv2.polylines(frame, [pts_to_draw], False, color, 12, lineType=cv2.LINE_AA)
            frames.append(frame)
            
        cv2.polylines(current_img, [curve_pts], False, color, 12, lineType=cv2.LINE_AA)
    
    while len(frames) < total_frames:
        frames.append(current_img.copy())
        
    writer = imageio.get_writer('/app/output/video.mp4', fps=fps, codec='libx264', pixelformat='yuv420p')
    for f in frames:
        writer.append_data(f)
    writer.close()

if __name__ == "__main__":
    main()
