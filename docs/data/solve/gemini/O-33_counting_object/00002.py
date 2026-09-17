import cv2
import numpy as np
import imageio
import os

def solve():
    img = cv2.imread('/app/first_frame.png')
    if img is None:
        raise ValueError("Could not read first_frame.png")
    
    # Find objects (black outlines)
    diff = np.abs(img.astype(int) - 0)
    mask = np.all(diff < 50, axis=2).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    valid_contours = [c for c in contours if cv2.contourArea(c) > 50]
    # Sort systematically (e.g., left to right, then top to bottom)
    valid_contours.sort(key=lambda c: (cv2.boundingRect(c)[0] // 50, cv2.boundingRect(c)[1]))
    
    N = len(valid_contours)
    
    fps = 16
    total_frames = 35
    
    # We want to pace the action.
    # Frame 0, 1: pause
    # Frame 2 to end_anim-1: animate counting
    # Frame end_anim to 34: display "Count: N"
    
    initial_pause = 2
    final_hold = 12
    anim_frames = total_frames - initial_pause - final_hold
    
    if N > 0:
        frames_per_object = max(1, anim_frames // N)
    else:
        frames_per_object = 1
        
    writer = imageio.get_writer('/app/output/video.mp4', fps=fps, codec='libx264', pixelformat='yuv420p')
    
    for f in range(total_frames):
        frame = img.copy()
        
        # Determine how many objects are fully drawn, and if one is currently being drawn
        if f < initial_pause:
            # Nothing drawn yet
            pass
        elif f >= initial_pause + N * frames_per_object:
            # All drawn
            for c in valid_contours:
                cv2.drawContours(frame, [c], -1, (0, 0, 255), 5)
                
            # Display text in the final frames
            text = f"Count: {N}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 3
            thickness = 6
            (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
            tx = (frame.shape[1] - tw) // 2
            ty = (frame.shape[0] + th) // 2
            cv2.putText(frame, text, (tx, ty), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
        else:
            # Mid-animation
            t = f - initial_pause
            current_obj = t // frames_per_object
            sub_t = t % frames_per_object
            
            # Draw fully completed objects
            for i in range(current_obj):
                cv2.drawContours(frame, [valid_contours[i]], -1, (0, 0, 255), 5)
                
            # Draw currently animating object
            c = valid_contours[current_obj]
            # sub_t goes from 0 to frames_per_object - 1
            # We want fraction to go from >0 to 1
            fraction = (sub_t + 1) / frames_per_object
            num_points = max(1, int(len(c) * fraction))
            pts = c[:num_points]
            
            if num_points == len(c):
                cv2.drawContours(frame, [c], -1, (0, 0, 255), 5)
            else:
                cv2.polylines(frame, [pts], False, (0, 0, 255), 5)
                
        # Convert BGR to RGB for imageio
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_rgb)
        
    writer.close()

if __name__ == '__main__':
    solve()
