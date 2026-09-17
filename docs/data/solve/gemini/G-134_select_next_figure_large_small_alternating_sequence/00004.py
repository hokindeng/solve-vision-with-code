import cv2
import numpy as np
import os
import imageio

def get_frames():
    img = cv2.imread('/app/first_frame.png')
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    frames = []
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    blue = (0, 100, 255)
    red = (255, 0, 0)
    thickness = 2
    
    # Align text baselines
    text_y = 430
    
    seq = [
        {"rect": (221, 317, 41, 41), "text": "SMALL", "text_pos": (206, text_y)},
        {"rect": (376, 292, 91, 91), "text": "LARGE", "text_pos": (383, text_y)},
        {"rect": (582, 317, 41, 41), "text": "SMALL", "text_pos": (567, text_y)}
    ]
    
    pred = {"rect": (737, 292, 91, 91), "text": "LARGE?", "text_pos": (740, text_y)}
    
    choice_circle = {"center": (61+211//2, 752+205//2), "radius": 110}
    
    def draw_box(target_img, rect, progress, color):
        x, y, w, h = rect
        total_len = 2*w + 2*h
        draw_len = int(progress * total_len)
        
        pts = [(x, y), (x+w, y), (x+w, y+h), (x, y+h), (x, y)]
        drawn = 0
        for i in range(4):
            pt1 = pts[i]
            pt2 = pts[i+1]
            seg_len = w if i%2==0 else h
            
            if draw_len <= drawn:
                break
            
            if draw_len >= drawn + seg_len:
                cv2.line(target_img, pt1, pt2, color, thickness)
                drawn += seg_len
            else:
                rem = draw_len - drawn
                ratio = rem / seg_len
                pt3 = (int(pt1[0] + (pt2[0]-pt1[0])*ratio), int(pt1[1] + (pt2[1]-pt1[1])*ratio))
                cv2.line(target_img, pt1, pt3, color, thickness)
                break
                
    def draw_dashed_box(target_img, rect, progress, color):
        x, y, w, h = rect
        total_len = 2*w + 2*h
        draw_len = int(progress * total_len)
        
        dash_len = 10
        drawn = 0
        
        pts = [(x, y), (x+w, y), (x+w, y+h), (x, y+h), (x, y)]
        for i in range(4):
            pt1 = pts[i]
            pt2 = pts[i+1]
            seg_len = w if i%2==0 else h
            
            if draw_len <= drawn:
                break
            
            d = 0
            while d < seg_len:
                if drawn + d >= draw_len:
                    break
                
                start_d = d
                end_d = min(d + dash_len // 2, seg_len)
                
                if drawn + end_d > draw_len:
                    end_d = draw_len - drawn
                
                ratio1 = start_d / seg_len
                ratio2 = end_d / seg_len
                
                pA = (int(pt1[0] + (pt2[0]-pt1[0])*ratio1), int(pt1[1] + (pt2[1]-pt1[1])*ratio1))
                pB = (int(pt1[0] + (pt2[0]-pt1[0])*ratio2), int(pt1[1] + (pt2[1]-pt1[1])*ratio2))
                
                cv2.line(target_img, pA, pB, color, thickness)
                
                d += dash_len
            drawn += seg_len

    total_frames = 64
    
    for frame_idx in range(total_frames):
        f_img = img_rgb.copy()
        
        # Phase 1: Seq 1 (Frames 0-8)
        p1 = min(max((frame_idx - 0) / 8.0, 0.0), 1.0)
        if p1 > 0:
            draw_box(f_img, seq[0]["rect"], p1, blue)
            if p1 > 0.5:
                cv2.putText(f_img, seq[0]["text"], seq[0]["text_pos"], font, font_scale, blue, thickness)
                
        # Phase 2: Seq 2 (Frames 8-16)
        p2 = min(max((frame_idx - 8) / 8.0, 0.0), 1.0)
        if p2 > 0:
            draw_box(f_img, seq[1]["rect"], p2, blue)
            if p2 > 0.5:
                cv2.putText(f_img, seq[1]["text"], seq[1]["text_pos"], font, font_scale, blue, thickness)
                
        # Phase 3: Seq 3 (Frames 16-24)
        p3 = min(max((frame_idx - 16) / 8.0, 0.0), 1.0)
        if p3 > 0:
            draw_box(f_img, seq[2]["rect"], p3, blue)
            if p3 > 0.5:
                cv2.putText(f_img, seq[2]["text"], seq[2]["text_pos"], font, font_scale, blue, thickness)
                
        # Phase 4: Predict (Frames 24-36)
        p4 = min(max((frame_idx - 24) / 12.0, 0.0), 1.0)
        if p4 > 0:
            draw_dashed_box(f_img, pred["rect"], p4, blue)
            if p4 > 0.5:
                cv2.putText(f_img, pred["text"], pred["text_pos"], font, font_scale, blue, thickness)
                
        # Phase 5: Circle Choice (Frames 36-50)
        p5 = min(max((frame_idx - 36) / 14.0, 0.0), 1.0)
        if p5 > 0:
            angle = int(p5 * 360)
            cv2.ellipse(f_img, choice_circle["center"], (choice_circle["radius"], choice_circle["radius"]), 
                        0, 0, angle, red, 4)
                        
        frames.append(f_img)
        
    return frames

def make_video():
    frames = get_frames()
    os.makedirs('/app/output', exist_ok=True)
    out_path = '/app/output/video.mp4'
    writer = imageio.get_writer(out_path, fps=16, codec='libx264', macro_block_size=None, pixelformat='yuv420p')
    for f in frames:
        writer.append_data(f)
    writer.close()

if __name__ == '__main__':
    make_video()
