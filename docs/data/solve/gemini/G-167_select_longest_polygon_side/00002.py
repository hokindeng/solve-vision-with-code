import cv2
import numpy as np
import imageio
import math
import os

def main():
    os.makedirs('/app/output', exist_ok=True)

    bgr_img = cv2.imread('/app/first_frame.png')
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key=cv2.contourArea)

    epsilon = 0.001 * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, epsilon, True)

    vertices = [(int(pt[0][0]), int(pt[0][1])) for pt in approx]

    lengths = []
    for i in range(len(vertices)):
        p1 = vertices[i]
        p2 = vertices[(i+1)%len(vertices)]
        dist = math.hypot(p1[0]-p2[0], p1[1]-p2[1])
        lengths.append(dist)

    max_len_idx = lengths.index(max(lengths))
    p1_max = vertices[max_len_idx]
    p2_max = vertices[(max_len_idx+1)%len(vertices)]
    midpoint = (int(round((p1_max[0]+p2_max[0])/2)), int(round((p1_max[1]+p2_max[1])/2)))

    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', pixelformat='yuv420p')

    # Frame 0: original
    writer.append_data(rgb_img)

    highlight_color = (0, 255, 0) # Green for comparing
    thickness = 6

    # Frames 1-16: highlight each edge for 2 frames
    for i in range(len(vertices)):
        p1 = vertices[i]
        p2 = vertices[(i+1)%len(vertices)]
        
        frame = rgb_img.copy()
        cv2.line(frame, p1, p2, highlight_color, thickness)
        
        writer.append_data(frame)
        writer.append_data(frame)

    # Circle color: red (255, 0, 0 in RGB)
    circle_color = (255, 0, 0)

    # Frames 17-20: circle grows
    radii = [4, 8, 12, 16]
    for r in radii:
        frame = rgb_img.copy()
        cv2.circle(frame, midpoint, r, circle_color, -1)
        writer.append_data(frame)

    # Frames 21-24: hold final
    final_frame = rgb_img.copy()
    cv2.circle(final_frame, midpoint, 16, circle_color, -1)
    for _ in range(4):
        writer.append_data(final_frame)

    writer.close()

if __name__ == '__main__':
    main()
