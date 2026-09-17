import cv2
import numpy as np
import imageio
import os

def main():
    os.makedirs('/app/output', exist_ok=True)

    img = cv2.imread('/app/first_frame.png')
    
    # 1. Find the large circular region
    black_mask = cv2.inRange(img, np.array([0,0,0]), np.array([0,0,0]))
    contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    circle_center = None
    circle_radius = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 10000:
            approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
            # A circle will have many vertices, whereas squares have 4
            if len(approx) > 6:
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                circle_center = (int(x), int(y))
                circle_radius = int(radius)
                break
                
    if circle_center is None:
        raise ValueError("Could not find the circular region.")

    # 2. Find all shapes with gray borders (this targets the objects)
    gray_mask = cv2.inRange(img, np.array([80,80,80]), np.array([80,80,80]))
    contours, _ = cv2.findContours(gray_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    all_triangles_pixels = []

    for cnt in contours:
        M = cv2.moments(cnt)
        if M['m00'] > 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            
            # Check if the shape is inside the large circle
            dist = np.hypot(cx - circle_center[0], cy - circle_center[1])
            if dist < circle_radius:
                approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
                if len(approx) == 3: # It's a triangle
                    # Isolate exactly the border pixels of this shape
                    full_mask = np.zeros_like(gray_mask)
                    cv2.drawContours(full_mask, [cnt], -1, 255, thickness=cv2.FILLED)
                    isolated_border = cv2.bitwise_and(gray_mask, full_mask)
                    
                    pixels = np.argwhere(isolated_border > 0)
                    
                    # Sort pixels by angle to trace the outline continuously
                    def get_angle(p):
                        return np.arctan2(p[0] - cy, p[1] - cx)
                    
                    pixels = sorted(pixels, key=get_angle)
                    all_triangles_pixels.append(pixels)

    # 3. Generate the video
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', pixelformat='yuv420p', macro_block_size=None)

    num_frames = 40
    for i in range(num_frames):
        frame = img.copy()
        
        for pixels in all_triangles_pixels:
            count = int(len(pixels) * i / (num_frames - 1))
            for p in pixels[:count]:
                frame[p[0], p[1]] = [0, 255, 0] # BGR format
                
        # Convert BGR to RGB for imageio
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_rgb)

    writer.close()

if __name__ == "__main__":
    main()
