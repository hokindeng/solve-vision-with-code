import cv2
import numpy as np
import imageio
import os

def get_unique_contour(img):
    """
    Finds the contour of the unique shape based on its geometrical features.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Background is assumed to be white, shapes are dark
    ret, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter small noise
    contours = [c for c in contours if cv2.contourArea(c) > 100]
    
    features = []
    for c in contours:
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = float(w) / h
        extent = float(area) / (w * h)
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0
        features.append(np.array([area, perimeter, aspect_ratio, extent, solidity]))
        
    features = np.array(features)
    
    # Normalize features for distance calculation
    means = np.mean(features, axis=0)
    stds = np.std(features, axis=0)
    stds[stds == 0] = 1.0  # Avoid division by zero
    
    norm_features = (features - means) / stds
    
    # The unique shape will have the highest total distance to all other shapes
    distances = np.zeros(len(contours))
    for i in range(len(contours)):
        dist = 0
        for j in range(len(contours)):
            if i != j:
                dist += np.linalg.norm(norm_features[i] - norm_features[j])
        distances[i] = dist
        
    unique_idx = np.argmax(distances)
    return contours[unique_idx]

def solve():
    img_path = '/app/first_frame.png'
    out_path = '/app/output/video.mp4'
    
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read {img_path}")
        
    unique_contour = get_unique_contour(img)
    
    # Get the bounding circle for the unique shape
    (x, y), radius = cv2.minEnclosingCircle(unique_contour)
    center = (int(x), int(y))
    radius = int(radius) + 20  # Add some padding to the circle
    
    # Video parameters
    num_frames = 60
    fps = 16
    
    writer = imageio.get_writer(
        out_path, 
        fps=fps, 
        codec='libx264', 
        format='FFMPEG', 
        macro_block_size=None, 
        pixelformat='yuv420p'
    )
    
    # Animation timings
    start_delay = 10
    draw_frames = 40
    
    for i in range(num_frames):
        frame = img.copy()
        
        # Step-by-step drawing of the circle
        if i >= start_delay:
            # Calculate how much of the circle to draw (0.0 to 1.0)
            progress = min(1.0, (i - start_delay) / (draw_frames - 1))
            angle = int(progress * 360)
            
            if angle > 0:
                # Draw arc from top (-90 degrees) clockwise
                cv2.ellipse(frame, center, (radius, radius), -90, 0, angle, (0, 0, 255), 8, cv2.LINE_AA)
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_rgb)
        
    writer.close()

if __name__ == '__main__':
    os.makedirs('/app/output', exist_ok=True)
    solve()
