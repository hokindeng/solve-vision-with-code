import cv2
import numpy as np
import imageio

def get_contours(img):
    # The background color is a uniform gray
    bg_color = np.array([240, 240, 240])
    mask_bg = cv2.inRange(img, bg_color, bg_color)
    mask = cv2.bitwise_not(mask_bg)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter out any tiny noise contours just in case
    valid_contours = []
    for c in contours:
        if cv2.contourArea(c) > 100:
            valid_contours.append(c)
            
    # Sort systematically: top to bottom, left to right
    # Group by rows of 100 pixels
    def get_sort_key(c):
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            x, y, w, h = cv2.boundingRect(c)
            cX = x + w//2
            cY = y + h//2
        row = cY // 100
        return (row, cX)

    valid_contours.sort(key=get_sort_key)
    return valid_contours

def main():
    img = cv2.imread('/app/first_frame.png')
    
    contours = get_contours(img)
    N = len(contours)
    
    fps = 16
    total_frames = 70
    
    writer = imageio.get_writer('/app/output/video.mp4', fps=fps, codec='libx264', pixelformat='yuv420p')
    
    # We have N objects. We want to pace them out.
    start_frame = 5
    frames_per_obj = 5
    text_frame = start_frame + N * frames_per_obj + 5
    
    for i in range(total_frames):
        frame = img.copy()
        
        # Determine how many objects are currently counted
        if i < start_frame:
            counted = 0
        else:
            counted = (i - start_frame) // frames_per_obj + 1
            if counted > N:
                counted = N
                
        # Draw highlights (thick red border)
        for j in range(counted):
            cv2.drawContours(frame, [contours[j]], -1, (0, 0, 255), 6)
            
        # Draw text
        if i >= text_frame:
            text = f"Count: {N}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 3
            thickness = 6
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            
            text_x = (frame.shape[1] - text_size[0]) // 2
            text_y = (frame.shape[0] + text_size[1]) // 2
            
            # White outline
            cv2.putText(frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness + 12, cv2.LINE_AA)
            # Black text
            cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
            
        # Convert BGR to RGB for imageio
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_rgb)
        
    writer.close()

if __name__ == '__main__':
    main()
