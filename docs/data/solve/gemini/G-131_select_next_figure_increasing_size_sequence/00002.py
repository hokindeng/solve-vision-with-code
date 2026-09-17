import cv2
import numpy as np
import imageio
import os

def solve():
    img = cv2.imread('/app/first_frame.png')
    if img is None:
        raise ValueError("Image not found")
        
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (0, 150, 255) # BGR: orange-ish
    thickness = 2
    
    os.makedirs('/app/output', exist_ok=True)
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', pixelformat='yuv420p', macro_block_size=None)
    
    # Pre-define drawing functions
    def draw_w1(img): cv2.putText(img, "27", (225, 310), font, 0.8, color, thickness)
    def draw_w2(img): cv2.putText(img, "43", (405, 300), font, 0.8, color, thickness)
    def draw_w3(img): cv2.putText(img, "60", (585, 290), font, 0.8, color, thickness)
    
    def draw_a1(img):
        cv2.arrowedLine(img, (270, 337), (385, 337), color, 2, tipLength=0.1)
        cv2.putText(img, "+16.5", (290, 325), font, 0.6, color, 2)
        
    def draw_a2(img):
        cv2.arrowedLine(img, (460, 337), (555, 337), color, 2, tipLength=0.1)
        cv2.putText(img, "+16.5", (475, 325), font, 0.6, color, 2)
        
    def draw_a3(img):
        cv2.arrowedLine(img, (650, 337), (745, 337), color, 2, tipLength=0.1)
        cv2.putText(img, "+16.5", (665, 325), font, 0.6, color, 2)
        
    def draw_next(img):
        cv2.putText(img, "Next: 76.5", (760, 345), font, 0.9, color, 3)
        
    elements = [
        (4, 5, draw_w1),
        (9, 5, draw_w2),
        (14, 5, draw_w3),
        (19, 5, draw_a1),
        (24, 5, draw_a2),
        (29, 5, draw_a3),
        (34, 5, draw_next)
    ]
    
    for f in range(60):
        frame = img.copy()
        
        # Draw fade elements
        for start, dur, func in elements:
            if f >= start:
                alpha = min(1.0, (f - start) / dur)
                if alpha < 1.0:
                    overlay = frame.copy()
                    func(overlay)
                    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
                else:
                    func(frame)
                    
        # Draw circle incrementally
        circ_start = 42
        circ_dur = 10
        if f >= circ_start:
            progress = min(1.0, (f - circ_start) / circ_dur)
            angle = progress * 360
            if angle > 0:
                cv2.ellipse(frame, (626, 854), (115, 115), 0, 0, angle, (0, 0, 255), 5)
                
        # convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_rgb)
        
    writer.close()

if __name__ == "__main__":
    solve()
