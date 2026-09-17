import cv2
import numpy as np
import imageio

def solve():
    # 1. Read the first frame
    img = cv2.imread('/app/first_frame.png')
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 2. Identify the points
    # Points are pure black pixels
    black_mask = np.all(img == [0, 0, 0], axis=-1).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(black_mask, connectivity=8)

    # Overlap color in BGR
    overlap_color = [69, 200, 97] 
    
    target_points = []
    for i in range(1, num_labels):
        # Create a mask for this point
        pt_mask = (labels == i).astype(np.uint8)
        
        # Dilate to get the boundary of the point
        dilated = cv2.dilate(pt_mask, np.ones((7,7), np.uint8))
        boundary = dilated - pt_mask
        boundary_pixels = img[boundary == 1]
        
        unique_colors, counts = np.unique(boundary_pixels, axis=0, return_counts=True)
        
        # Check if the point is fully inside the overlap region (all boundary pixels are the overlap color)
        is_fully_inside = True
        for c in unique_colors:
            if not np.array_equal(c, overlap_color):
                is_fully_inside = False
                break
                
        if is_fully_inside:
            target_points.append(centroids[i])

    # Sort points by x coordinate to have a consistent visual order (left to right)
    target_points.sort(key=lambda p: p[0])

    # 3. Animate the solution step by step
    fps = 16
    total_frames = 37
    
    writer = imageio.get_writer(
        '/app/output/video.mp4', 
        fps=fps, 
        codec='libx264', 
        pixelformat='yuv420p', 
        macro_block_size=None
    )

    frames_per_point = (total_frames - 1) // len(target_points)
    radius = 20
    thickness = 3
    color = (255, 0, 0) # RGB red

    for frame_idx in range(total_frames):
        frame = img_rgb.copy()
        
        if frame_idx > 0:
            for i, pt in enumerate(target_points):
                cx, cy = int(pt[0]), int(pt[1])
                
                start_frame_for_pt = 1 + i * frames_per_point
                end_frame_for_pt = start_frame_for_pt + frames_per_point - 1
                
                if frame_idx >= start_frame_for_pt:
                    # Calculate progress (0.0 to 1.0) for the drawing animation
                    if frame_idx >= end_frame_for_pt:
                        progress = 1.0
                    else:
                        progress = (frame_idx - start_frame_for_pt + 1) / frames_per_point
                    
                    angle = int(360 * progress)
                    
                    if angle > 0:
                        cv2.ellipse(frame, (cx, cy), (radius, radius), 0, 0, angle, color, thickness, cv2.LINE_AA)
                        
        writer.append_data(frame)

    writer.close()

if __name__ == '__main__':
    solve()
