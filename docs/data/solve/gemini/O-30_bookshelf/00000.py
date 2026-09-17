import cv2
import numpy as np
import imageio
import os

def ease_in_out(t):
    return (1 - np.cos(np.pi * t)) / 2

def solve():
    img = cv2.imread('/app/first_frame.png')
    clean_bg = img.copy()

    books = [
        {"rect": (710, 373, 27, 140), "target_x": 166, "lift": 250, "start_f": 0, "dur_f": 12},
        {"rect": (806, 370, 27, 143), "target_x": 262, "lift": 200, "start_f": 4, "dur_f": 12},
        {"rect": (774, 299, 27, 214), "target_x": 550, "lift": 150, "start_f": 8, "dur_f": 12},
        {"rect": (742, 275, 27, 238), "target_x": 614, "lift": 100, "start_f": 12, "dur_f": 12}
    ]

    patches = []
    for b in books:
        x, y, w, h = b["rect"]
        patches.append(img[y:y+h, x:x+w].copy())
        # Clean the background where the books initially were
        for j in range(y, y+h):
            for i in range(x, x+w):
                if j < 512:
                    clean_bg[j, i] = [255, 255, 255]
                else:
                    clean_bg[j, i] = [49, 71, 102]

    frames = []
    total_frames = 25

    for f in range(total_frames):
        frame = clean_bg.copy()
        
        # Draw books
        for i, b in enumerate(books):
            x, y, w, h = b["rect"]
            tx = b["target_x"]
            
            if f <= b["start_f"]:
                t = 0
            elif f >= b["start_f"] + b["dur_f"]:
                t = 1
            else:
                t = (f - b["start_f"]) / b["dur_f"]
                
            x_t = ease_in_out(t)
            y_t = np.sin(np.pi * t)
            
            cur_x = int(x + (tx - x) * x_t)
            cur_y = int(y - b["lift"] * y_t)
            
            # Since the books are solid rectangles, we can just slice assignment
            # But need to handle boundaries just in case, although they shouldn't hit edges
            # Actually, np array assignment is faster and easier.
            # We just need to make sure we don't go out of bounds.
            y1 = max(0, cur_y)
            y2 = min(1024, cur_y + h)
            x1 = max(0, cur_x)
            x2 = min(1024, cur_x + w)
            
            patch_y1 = y1 - cur_y
            patch_y2 = h - (cur_y + h - y2)
            patch_x1 = x1 - cur_x
            patch_x2 = w - (cur_x + w - x2)
            
            if y2 > y1 and x2 > x1:
                frame[y1:y2, x1:x2] = patches[i][patch_y1:patch_y2, patch_x1:patch_x2]
                        
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Add a final hold frame? No, 25 frames is enough and ends on the exact final state.

    os.makedirs('/app/output', exist_ok=True)
    
    # Save video with required specs: H.264, yuv420p, 16fps
    imageio.mimwrite('/app/output/video.mp4', frames, fps=16, macro_block_size=None,
                     codec='libx264', pixelformat='yuv420p')

if __name__ == '__main__':
    solve()
