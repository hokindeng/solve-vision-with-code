import cv2
import numpy as np
import imageio
import os

def get_dense_polygon(approx, num_points=1000):
    pts = approx.reshape(-1, 2)
    # close the loop
    pts = np.vstack([pts, pts[0]])
    n = len(pts)
    lengths = np.zeros(n)
    for i in range(1, n):
        lengths[i] = lengths[i-1] + np.linalg.norm(pts[i] - pts[i-1])
    
    total_length = lengths[-1]
    
    dense_pts = []
    for dist in np.linspace(0, total_length, num_points):
        for i in range(1, n):
            if dist <= lengths[i] + 1e-7:
                prev = pts[i-1]
                curr = pts[i]
                seg_len = lengths[i] - lengths[i-1]
                if seg_len == 0:
                    new_pt = prev
                else:
                    ratio = (dist - lengths[i-1]) / seg_len
                    new_pt = prev + ratio * (curr - prev)
                dense_pts.append(new_pt)
                break
    return np.array(dense_pts, dtype=np.int32).reshape(-1, 1, 2)

def main():
    # Read the first frame
    img = cv2.imread('/app/first_frame.png')
    if img is None:
        raise FileNotFoundError("Could not read /app/first_frame.png")
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # The background is white (255), shapes are colored. 
    # Threshold to find shapes
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    trapezoids = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # Filter for typical shape sizes
        if 10 < w < 100 and 10 < h < 100:
            # Check if strictly in the square region (approx bounding box of the square region)
            # The square region is around x=567, y=324, w=377, h=377
            if x >= 567 and y >= 324 and x+w <= 944 and y+h <= 701:
                epsilon = 0.04 * cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, epsilon, True)
                
                if len(approx) == 4:
                    pts = approx.reshape(4, 2)
                    # Compute lengths of the 4 sides
                    s0 = np.linalg.norm(pts[0] - pts[1])
                    s1 = np.linalg.norm(pts[1] - pts[2])
                    s2 = np.linalg.norm(pts[2] - pts[3])
                    s3 = np.linalg.norm(pts[3] - pts[0])
                    
                    # A parallelogram (like square/rectangle) has opposite sides equal.
                    # If opposite sides differ significantly, it's a trapezoid (in this context).
                    if abs(s0 - s2) > 5 or abs(s1 - s3) > 5:
                        trapezoids.append(approx)
                        
    # Generate dense polygons for drawing smoothly
    dense_polys = [get_dense_polygon(t, num_points=1500) for t in trapezoids]
    
    # Video generation parameters
    num_frames = 40
    fps = 16
    os.makedirs('/app/output', exist_ok=True)
    out_path = '/app/output/video.mp4'
    
    writer = imageio.get_writer(
        out_path, 
        fps=fps, 
        codec='libx264', 
        format='FFMPEG', 
        pixelformat='yuv420p',
        macro_block_size=None
    )
    
    for f in range(num_frames):
        frame = img.copy()
        fraction = f / (num_frames - 1)
        
        for poly in dense_polys:
            num_pts = len(poly)
            slice_idx = int(fraction * num_pts)
            if slice_idx > 1:
                # Draw the progressive outline
                cv2.polylines(frame, [poly[:slice_idx]], isClosed=False, color=(0, 255, 0), thickness=3)
        
        # Convert BGR to RGB for imageio
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_rgb)
        
    writer.close()

if __name__ == '__main__':
    main()
