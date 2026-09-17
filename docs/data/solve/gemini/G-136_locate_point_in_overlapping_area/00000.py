import cv2
import numpy as np
import subprocess
import os

def get_points(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY_INV)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    return num_labels, labels, centroids

def main():
    img = cv2.imread('/app/first_frame.png')
    
    # Dynamically find the overlap color if possible, but fallback to hardcoded
    colors, counts = np.unique(img.reshape(-1, 3), axis=0, return_counts=True)
    # Exclude background (largest) and points (assumed black or smallest)
    # Actually, we know the exact color for this image
    overlap_color = np.array([96, 149, 82])
    
    # verify if overlap_color is in the image, else try to find it
    if not any(np.array_equal(overlap_color, c) for c in colors):
        # find black color (points)
        non_bg_colors = [c for c in colors if not (np.all(c == 255) or np.all(c == 0))]
        # we expect 3 colors: shape1, shape2, overlap. 
        # The overlap area usually has the smallest area of the 3? Not necessarily.
        # But we'll sort by count.
        if len(non_bg_colors) == 3:
            # just guess the one with smallest count is overlap
            overlap_color = sorted([(c, count) for c, count in zip(colors, counts) if not (np.all(c==255) or np.all(c==0))], key=lambda x: x[1])[0][0]

    num_labels, labels, centroids = get_points(img)
    
    inside_points = []
    for i in range(1, num_labels):
        point_mask = (labels == i).astype(np.uint8)
        # Dilate by a small amount to get the immediate surroundings
        kernel = np.ones((5,5), np.uint8)
        dilated = cv2.dilate(point_mask, kernel, iterations=1)
        boundary_mask = dilated - point_mask
        boundary_pixels = img[boundary_mask == 1]
        unique_boundary_colors = np.unique(boundary_pixels, axis=0)
        
        is_inside = True
        for c in unique_boundary_colors:
            if not np.array_equal(c, overlap_color):
                is_inside = False
                break
        
        if is_inside:
            inside_points.append(i)
            
    print(f"Found inside points: {inside_points}")
    
    centers = []
    for p in inside_points:
        c = centroids[p]
        centers.append((int(c[0]), int(c[1])))
    
    # Generate 37 frames
    num_frames = 37
    fps = 16
    out_dir = '/app/output'
    os.makedirs(out_dir, exist_ok=True)
    
    frames_dir = '/tmp/frames'
    os.makedirs(frames_dir, exist_ok=True)
    
    N = len(centers)
    start_pause = 5
    end_pause = 5
    draw_frames = num_frames - start_pause - end_pause
    if N > 0:
        frames_per_point = max(1, draw_frames // N)
    else:
        frames_per_point = 1
        
    for t in range(num_frames):
        frame = img.copy()
        
        for i, center in enumerate(centers):
            start_t = start_pause + i * frames_per_point
            end_t = start_t + frames_per_point - 1
            
            if t >= start_t and t <= end_t:
                # animate
                progress = (t - start_t) / max(1, (end_t - start_t))
                end_angle = int(progress * 360)
                cv2.ellipse(frame, center, (20, 20), -90, 0, end_angle, (0, 0, 255), 4, cv2.LINE_AA)
            elif t > end_t:
                # fully drawn
                cv2.ellipse(frame, center, (20, 20), -90, 0, 360, (0, 0, 255), 4, cv2.LINE_AA)
                
        cv2.imwrite(f"{frames_dir}/frame_{t:04d}.png", frame)
        
    out_path = f"{out_dir}/video.mp4"
    if os.path.exists(out_path):
        os.remove(out_path)
        
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", f"{frames_dir}/frame_%04d.png",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        out_path
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("Video generated successfully.")

if __name__ == '__main__':
    main()
