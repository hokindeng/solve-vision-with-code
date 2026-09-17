import cv2
import numpy as np
import imageio.v2 as imageio
import os

def main():
    # Read first frame
    img_bgr = cv2.imread('/app/first_frame.png')
    if img_bgr is None:
        raise ValueError("Could not read /app/first_frame.png")
    
    # We will work in RGB for imageio
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Find black pixels to identify outlines
    # In BGR, black is [0,0,0]. We give a small tolerance.
    black_mask = cv2.inRange(img_bgr, np.array([0,0,0]), np.array([50,50,50]))
    
    # Find external contours of the black outlines
    contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours systematically (e.g., left to right)
    def get_center(c):
        M = cv2.moments(c)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            x, y, w, h = cv2.boundingRect(c)
            cx = x + w // 2
            cy = y + h // 2
        return cx, cy

    contours = sorted(contours, key=lambda c: get_center(c)[0] * 10000 + get_center(c)[1])
    
    N = len(contours)
    
    # Frame allocation
    total_frames = 40
    fps = 16
    
    initial_frames = 4
    final_frames = 12
    counting_frames = total_frames - initial_frames - final_frames
    frames_per_object = counting_frames // N if N > 0 else 0
    
    os.makedirs('/app/output', exist_ok=True)
    writer = imageio.get_writer('/app/output/video.mp4', fps=fps, codec='libx264', pixelformat='yuv420p', macro_block_size=None)
    
    # Helper to highlight objects up to index `count_idx`
    def get_frame(count_idx, show_text=False):
        frame = img_rgb.copy()
        
        # Highlight objects 0 to count_idx-1
        for i in range(count_idx):
            c = contours[i]
            # Create a mask for this object
            mask = np.zeros_like(black_mask)
            cv2.drawContours(mask, [c], -1, 255, -1)
            # Find black pixels inside this object
            obj_black = cv2.bitwise_and(black_mask, mask)
            
            # Change these pixels to red in the frame
            # Red is (255, 0, 0) in RGB
            frame[obj_black == 255] = [255, 0, 0]
            
        if show_text:
            text = f"Count: {N}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 3
            thickness = 6
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            
            text_x = (frame.shape[1] - text_size[0]) // 2
            text_y = (frame.shape[0] + text_size[1]) // 2
            
            # Draw text with outline for better visibility
            cv2.putText(frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness + 4, cv2.LINE_AA)
            cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
            
        return frame

    frame_idx = 0
    
    # Initial pause
    for _ in range(initial_frames):
        writer.append_data(get_frame(0, False))
        frame_idx += 1
        
    # Counting objects
    for i in range(1, N + 1):
        for _ in range(frames_per_object):
            writer.append_data(get_frame(i, False))
            frame_idx += 1
            
    # Final state
    remaining = total_frames - frame_idx
    for _ in range(remaining):
        writer.append_data(get_frame(N, True))
        frame_idx += 1
        
    writer.close()

if __name__ == '__main__':
    main()
