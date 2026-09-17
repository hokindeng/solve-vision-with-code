import cv2
import numpy as np
import imageio

def draw_fade(frame, element_func, start_f, end_f, current_f):
    if current_f < start_f:
        return
    elif start_f <= current_f <= end_f:
        alpha = (current_f - start_f) / float(end_f - start_f)
        overlay = frame.copy()
        element_func(overlay)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    else:
        element_func(frame)

def main():
    base_img = cv2.imread('first_frame.png', cv2.IMREAD_COLOR)
    
    num_frames = 60
    fps = 16
    frames = []
    
    box1 = ((223-10, 319-10), (223+37+10, 319+37+10), (0, 165, 255), 6)
    box2 = ((367-10, 283-10), (367+109+10, 283+109+10), (0, 165, 255), 6)
    box3 = ((584-10, 319-10), (584+37+10, 319+37+10), (0, 165, 255), 6)
    box4 = ((717, 272), (845, 400), (0, 165, 255), 6)
    
    circle_center = (751 + 211//2, 752 + 205//2)
    circle_radius = int(max(211, 205) / 2 * 1.1)
    circle_color = (0, 0, 255)
    circle_thickness = 6
    
    for i in range(num_frames):
        frame = base_img.copy()
        
        draw_fade(frame, lambda img: cv2.rectangle(img, box1[0], box1[1], box1[2], box1[3]), 4, 10, i)
        draw_fade(frame, lambda img: cv2.rectangle(img, box2[0], box2[1], box2[2], box2[3]), 13, 19, i)
        draw_fade(frame, lambda img: cv2.rectangle(img, box3[0], box3[1], box3[2], box3[3]), 22, 28, i)
        draw_fade(frame, lambda img: cv2.rectangle(img, box4[0], box4[1], box4[2], box4[3]), 31, 37, i)
        draw_fade(frame, lambda img: cv2.circle(img, circle_center, circle_radius, circle_color, circle_thickness), 40, 46, i)
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    out_path = '/app/output/video.mp4'
    writer = imageio.get_writer(out_path, fps=fps, codec='libx264', macro_block_size=None, quality=8, pixelformat='yuv420p')
    for f in frames:
        writer.append_data(f)
    writer.close()

if __name__ == '__main__':
    main()
