import cv2
import numpy as np
import os
import subprocess

def find_hollow_points(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    hollow_points = []
    if hierarchy is None:
        return hollow_points
        
    for i, c in enumerate(contours):
        parent = hierarchy[0][i][3]
        first_child = hierarchy[0][i][2]
        
        # Outer boundary
        if parent == -1:
            if first_child != -1:
                # Has a child, let's check its area to be sure it's a hole
                child_contour = contours[first_child]
                if cv2.contourArea(child_contour) > 50:
                    (x,y), r = cv2.minEnclosingCircle(c)
                    hollow_points.append(((int(x), int(y)), int(r)))

    return hollow_points

def main():
    os.makedirs('/app/output', exist_ok=True)
    first_frame_path = '/app/first_frame.png'
    
    hollow_points = find_hollow_points(first_frame_path)
    
    img = cv2.imread(first_frame_path)
    h, w = img.shape[:2]
    
    # Video settings
    fps = 16
    total_frames = 80
    
    # ffmpeg subprocess for writing video
    command = [
        'ffmpeg',
        '-y', # overwrite output
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{w}x{h}',
        '-pix_fmt', 'bgr24',
        '-r', str(fps),
        '-i', '-', # read from stdin
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '/app/output/video.mp4'
    ]
    
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    
    # Animation parameters
    start_frame = 10
    end_frame = 70
    
    for frame_idx in range(total_frames):
        frame = img.copy()
        
        if frame_idx >= start_frame:
            # Calculate how much of the circle to draw (0 to 360)
            if frame_idx >= end_frame:
                angle = 360.0
            else:
                progress = (frame_idx - start_frame) / float(end_frame - start_frame)
                angle = progress * 360.0
                
            # Draw for each hollow point
            for (cx, cy), r in hollow_points:
                # Radius for the red ring (slightly larger than the point's radius)
                ring_radius = r + 15
                thickness = 6
                color = (0, 0, 255) # Red in BGR
                
                # cv2.ellipse takes (center), (axes), angle, startAngle, endAngle, color, thickness
                cv2.ellipse(frame, (cx, cy), (ring_radius, ring_radius), 0, 0, int(angle), color, thickness)
                
        process.stdin.write(frame.tobytes())
        
    process.stdin.close()
    process.wait()

if __name__ == '__main__':
    main()
