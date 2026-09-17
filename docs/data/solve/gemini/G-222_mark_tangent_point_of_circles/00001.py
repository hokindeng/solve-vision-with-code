import cv2
import numpy as np
import imageio
import itertools

def main():
    # Load the first frame
    first_frame = cv2.imread('/app/first_frame.png')
    
    # Dynamically find the circles by looking at unique colors
    pixels = first_frame.reshape(-1, 3)
    # Using return_counts to only consider colors that appear frequently enough (ignore anti-aliasing if any)
    unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
    
    circles = []
    for col, count in zip(unique_colors, counts):
        if count < 100: 
            continue
        # Skip background color (assuming it's white or near-white)
        if np.all(col >= 240): 
            continue 
        
        mask = cv2.inRange(first_frame, col, col)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: 
            continue
        
        # Take the largest contour for this color
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)
        if M['m00'] == 0: 
            continue
            
        cx = M['m10'] / M['m00']
        cy = M['m01'] / M['m00']
        _, r = cv2.minEnclosingCircle(c)
        
        if r > 10: # Filter out noise
            circles.append({'center': np.array([cx, cy]), 'radius': r})

    # Find the pair of touching circles
    touching_pair = None
    min_diff = float('inf')
    for c1, c2 in itertools.combinations(circles, 2):
        dist = np.linalg.norm(c1['center'] - c2['center'])
        sum_r = c1['radius'] + c2['radius']
        diff = abs(dist - sum_r)
        
        # If the distance between centers is very close to the sum of their radii, they are touching
        if diff < 5 and diff < min_diff:
            min_diff = diff
            touching_pair = (c1, c2)

    if touching_pair:
        c1, c2 = touching_pair
        # Calculate the exact tangent point
        tangent_pt = c1['center'] + (c2['center'] - c1['center']) * (c1['radius'] / (c1['radius'] + c2['radius']))
        tx, ty = int(round(tangent_pt[0])), int(round(tangent_pt[1]))
    else:
        # Fallback just in case (should not happen based on our analysis)
        tx, ty = 537, 353

    # Video parameters
    output_path = '/app/output/video.mp4'
    fps = 16
    total_frames = 60
    radius = 40
    thickness = 5
    color = (0, 0, 0) # Black circle

    writer = imageio.get_writer(output_path, fps=fps, codec='libx264', pixelformat='yuv420p', macro_block_size=None)

    for i in range(total_frames):
        frame = first_frame.copy()
        
        # Calculate how much of the circle to draw
        # Start from -90 degrees (top) and go clockwise
        end_angle = (360 * i) / (total_frames - 1)
        
        if end_angle > 0:
            # cv2.ellipse signature: (img, center, axes, angle, startAngle, endAngle, color, thickness, lineType)
            # angle=0 means normal orientation.
            # startAngle=-90 means top.
            # endAngle= -90 + end_angle
            cv2.ellipse(frame, (tx, ty), (radius, radius), 0, -90, -90 + end_angle, color, thickness, cv2.LINE_AA)
            
        # Convert BGR to RGB for imageio
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_rgb)

    writer.close()

if __name__ == "__main__":
    main()
