import cv2
import numpy as np
import imageio
import os

def find_topmost_shape(img):
    colors, counts = np.unique(img.reshape(-1, img.shape[2]), axis=0, return_counts=True)
    # The background is usually the most frequent color
    bg_color = tuple(colors[np.argmax(counts)])
    
    colors = [tuple(c) for c in colors if tuple(c) != bg_color]
    
    masks = {c: np.all(img == np.array(c), axis=-1).astype(np.uint8) for c in colors}
    hulls = {}
    for c, m in masks.items():
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(cnts) == 0: continue
        # use largest contour
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        hull = cv2.convexHull(cnts[0])
        hm = np.zeros_like(m)
        cv2.drawContours(hm, [hull], -1, 1, -1)
        hulls[c] = hm

    kernel = np.ones((5,5), np.uint8)
    occludes = {c: [] for c in colors}
    for c1, m1 in masks.items():
        m1_dilated = cv2.dilate(m1, kernel, iterations=1)
        for c2, m2 in masks.items():
            if c1 == c2: continue
            
            # check if they touch
            if np.sum(m1_dilated * m2) == 0:
                continue
                
            overlap1 = np.sum(m1 * hulls.get(c2, np.zeros_like(m1)))
            overlap2 = np.sum(m2 * hulls.get(c1, np.zeros_like(m2)))
            if overlap1 > overlap2 + 10:
                occludes[c1].append(c2)

    unobscured = []
    for c in colors:
        is_occluded = False
        for other in colors:
            if other == c: continue
            if c in occludes[other]:
                is_occluded = True
        if not is_occluded:
            unobscured.append(c)
            
    if len(unobscured) == 0:
        unobscured = colors
    
    best_c = None
    best_ratio = -1
    for c in unobscured:
        cnts, _ = cv2.findContours(masks[c], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: continue
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        area = cv2.contourArea(cnts[0])
        hull = cv2.convexHull(cnts[0])
        hull_area = cv2.contourArea(hull)
        ratio = area / hull_area if hull_area > 0 else 0
        if ratio > best_ratio:
            best_ratio = ratio
            best_c = c
            
    return best_c

def main():
    img = cv2.imread('/app/first_frame.png')
    if img is None:
        raise FileNotFoundError("Could not find /app/first_frame.png")
        
    top_color = find_topmost_shape(img)
    if top_color is None:
        raise ValueError("No shapes found")
        
    mask = np.all(img == np.array(top_color), axis=-1).astype(np.uint8)
    
    # We use CHAIN_APPROX_NONE to get all points for smooth drawing
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    cnt = contours[0]
    
    # Close the loop
    cnt = np.vstack([cnt, [cnt[0]]])
    total_points = len(cnt)
    
    num_frames = 40
    fps = 16
    
    os.makedirs('/app/output', exist_ok=True)
    writer = imageio.get_writer('/app/output/video.mp4', fps=fps, codec='libx264', pixelformat='yuv420p')
    
    for f in range(num_frames):
        frame = img.copy()
        
        # Calculate how many points to draw
        # f goes from 0 to num_frames - 1
        points_to_draw = int(np.round((f / (num_frames - 1)) * total_points))
        
        if points_to_draw > 1:
            # cv2.polylines needs a list of paths
            path = cnt[:points_to_draw]
            cv2.polylines(frame, [path], isClosed=False, color=(0, 0, 255), thickness=5)
            
        # Convert BGR to RGB for imageio
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_rgb)
        
    writer.close()

if __name__ == '__main__':
    main()
