import cv2
import numpy as np
import imageio
import os

def solve():
    img = cv2.imread('/app/first_frame.png')
    if img is None:
        raise FileNotFoundError("Could not read /app/first_frame.png")

    frames = []
    
    total_frames = 60
    
    for f in range(total_frames):
        frame = img.copy()
        
        # Sequence step 1
        if f >= 2:
            cv2.rectangle(frame, (218, 314), (218+47, 314+47), (255, 0, 0), 2)
            cv2.putText(frame, "47", (225, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        # Sequence step 2
        if f >= 6:
            cv2.rectangle(frame, (388, 304), (388+67, 304+67), (255, 0, 0), 2)
            cv2.putText(frame, "67", (395, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        if f >= 10:
            cv2.arrowedLine(frame, (275, 337), (375, 337), (0, 150, 0), 2)
            cv2.putText(frame, "+20", (300, 325), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 150, 0), 2)
        # Sequence step 3
        if f >= 14:
            cv2.rectangle(frame, (559, 294), (559+87, 294+87), (255, 0, 0), 2)
            cv2.putText(frame, "87", (565, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        if f >= 18:
            cv2.arrowedLine(frame, (465, 337), (545, 337), (0, 150, 0), 2)
            cv2.putText(frame, "+20", (485, 325), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 150, 0), 2)
        # Target criteria
        if f >= 22:
            cv2.putText(frame, "Target:", (680, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            cv2.putText(frame, "Size 107", (680, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            cv2.putText(frame, "Same Color", (680, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        # Choice A
        if f >= 28:
            cv2.rectangle(frame, (113, 811), (113+107, 811+107), (0, 0, 255), 2)
            cv2.putText(frame, "Wrong Color", (110, 950), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.line(frame, (113, 811), (220, 918), (0, 0, 255), 3)
            cv2.line(frame, (220, 811), (113, 918), (0, 0, 255), 3)

        # Choice B
        if f >= 34:
            cv2.rectangle(frame, (341, 806), (341+111, 806+106), (0, 0, 255), 2)
            cv2.putText(frame, "Wrong Size", (340, 950), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.line(frame, (341, 806), (452, 912), (0, 0, 255), 3)
            cv2.line(frame, (452, 806), (341, 912), (0, 0, 255), 3)

        # Choice D
        if f >= 40:
            cv2.rectangle(frame, (813, 821), (813+87, 821+87), (0, 0, 255), 2)
            cv2.putText(frame, "Wrong Size", (810, 950), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.line(frame, (813, 821), (900, 908), (0, 0, 255), 3)
            cv2.line(frame, (900, 821), (813, 908), (0, 0, 255), 3)

        # Choice C
        if f >= 46:
            cv2.rectangle(frame, (573, 811), (573+107, 811+107), (0, 200, 0), 2)
            cv2.putText(frame, "Correct!", (580, 950), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
            
        if f >= 50:
            progress = min(1.0, (f - 50) / 9.0)
            angle = int(360 * progress)
            cv2.ellipse(frame, (626, 854), (110, 110), 0, 0, angle, (0, 0, 255), 5)
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        
    os.makedirs('/app/output', exist_ok=True)
    
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', pixelformat='yuv420p', macro_block_size=None)
    for frame_rgb in frames:
        writer.append_data(frame_rgb)
    writer.close()

if __name__ == "__main__":
    solve()
