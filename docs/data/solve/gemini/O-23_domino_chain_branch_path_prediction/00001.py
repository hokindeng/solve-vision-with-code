import cv2
import numpy as np
import imageio

def get_patch(img, rect, bg_color):
    x, y, w, h = rect
    patch = img[y:y+h, x:x+w]
    diff = np.abs(patch.astype(int) - bg_color).sum(axis=2)
    alpha = (diff > 10).astype(np.uint8) * 255
    rgba = cv2.cvtColor(patch, cv2.COLOR_BGR2BGRA)
    rgba[:,:,3] = alpha
    return rgba

def draw_domino_fast(frame, x, y, patch_rgba, angle):
    h, w = patch_rgba.shape[:2]
    cx, cy = x + w, y + h
    pad = int(np.ceil(np.hypot(w, h)))
    padded = np.zeros((h + 2*pad, w + 2*pad, 4), dtype=np.uint8)
    padded[pad:pad+h, pad:pad+w] = patch_rgba
    
    M = cv2.getRotationMatrix2D((pad + w, pad + h), angle, 1.0)
    rotated = cv2.warpAffine(padded, M, (padded.shape[1], padded.shape[0]))
    
    tl_x = cx - (pad + w)
    tl_y = cy - (pad + h)
    
    y1 = max(0, int(tl_y))
    y2 = min(frame.shape[0], int(tl_y + rotated.shape[0]))
    x1 = max(0, int(tl_x))
    x2 = min(frame.shape[1], int(tl_x + rotated.shape[1]))
    
    ry1 = y1 - int(tl_y)
    ry2 = ry1 + (y2 - y1)
    rx1 = x1 - int(tl_x)
    rx2 = rx1 + (x2 - x1)
    
    if y2 > y1 and x2 > x1:
        roi = frame[y1:y2, x1:x2]
        rot_rgb = rotated[ry1:ry2, rx1:rx2, :3]
        alpha = rotated[ry1:ry2, rx1:rx2, 3:4] / 255.0
        frame[y1:y2, x1:x2] = (1 - alpha) * roi + alpha * rot_rgb

def main():
    first_frame_path = '/app/first_frame.png'
    out_video_path = '/app/output/video.mp4'
    
    first_frame = cv2.imread(first_frame_path)
    if first_frame is None:
        raise ValueError(f"Could not read {first_frame_path}")
        
    bg_color = np.array([240, 245, 248])
    
    dominos = [
        {"name": "7A", "rect": (930, 293, 41, 105), "delay": 45, "max_angle": -90},
        {"name": "6A", "rect": (806, 314, 41, 105), "delay": 40, "max_angle": -75},
        {"name": "5A", "rect": (682, 335, 41, 105), "delay": 35, "max_angle": -75},
        {"name": "4A", "rect": (558, 356, 41, 105), "delay": 30, "max_angle": -75},
        {"name": "5B", "rect": (682, 562, 41, 105), "delay": 999, "max_angle": 0},
        {"name": "4B", "rect": (558, 542, 41, 105), "delay": 30, "max_angle": -90},
        {"name": "3", "rect": (434, 449, 41, 105), "delay": 25, "max_angle": -75},
        {"name": "2", "rect": (310, 449, 41, 105), "delay": 20, "max_angle": -75},
        {"name": "1", "rect": (186, 449, 41, 105), "delay": 15, "max_angle": -75},
        {"name": "0", "rect": (62, 449, 41, 105), "delay": 10, "max_angle": -75},
    ]
    
    # Sort for drawing: back-to-front by y_bottom, then left-to-right (right is behind left for falling)
    dominos.sort(key=lambda d: (d["rect"][1] + d["rect"][3], -d["rect"][0]))
    
    bg_img = first_frame.copy()
    for d in dominos:
        d["patch"] = get_patch(first_frame, d["rect"], bg_color)
        x, y, w, h = d["rect"]
        cv2.rectangle(bg_img, (x, y), (x+w, y+h), (240, 245, 248), -1) 
        
    writer = imageio.get_writer(out_video_path, fps=16, codec='libx264', macro_block_size=None, pixelformat='yuv420p')
    
    total_frames = 62
    speed = 10.0 # degrees per frame
    
    for f in range(total_frames):
        frame = bg_img.copy()
        
        for d in dominos:
            frames_falling = max(0, f - d["delay"])
            angle = min(frames_falling * speed, abs(d["max_angle"]))
            angle = -angle  # clockwise
            
            x, y, w, h = d["rect"]
            draw_domino_fast(frame, x, y, d["patch"], angle)
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_rgb)
        
    writer.close()

if __name__ == "__main__":
    main()
