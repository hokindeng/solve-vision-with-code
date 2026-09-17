import cv2
import numpy as np
import imageio
import os

def draw_bracket(img, x, w, y, text):
    color = (50, 50, 50)
    thick = 2
    cv2.line(img, (x, y), (x + w - 1, y), color, thick)
    cv2.line(img, (x, y - 5), (x, y + 5), color, thick)
    cv2.line(img, (x + w - 1, y - 5), (x + w - 1, y + 5), color, thick)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, 0.7, 2)[0]
    tx = x + w // 2 - text_size[0] // 2
    ty = y + 25
    cv2.putText(img, text, (tx, ty), font, 0.7, color, 2)

def draw_cross(img, cx, cy):
    color = (0, 0, 0)
    thick = 5
    s = 25
    cv2.line(img, (cx - s, cy - s), (cx + s, cy + s), color, thick)
    cv2.line(img, (cx + s, cy - s), (cx - s, cy + s), color, thick)

def draw_alpha(base_img, draw_func, alpha):
    if alpha <= 0: return base_img
    if alpha >= 1:
        overlay = base_img.copy()
        draw_func(overlay)
        return overlay
    overlay = base_img.copy()
    draw_func(overlay)
    return cv2.addWeighted(overlay, alpha, base_img, 1 - alpha, 0)

def main():
    original_img = cv2.imread('/app/first_frame.png')
    if original_img is None:
        print("Error: Could not read first_frame.png")
        return

    # Ensure output directory exists
    os.makedirs('/app/output', exist_ok=True)
    
    y_bracket = 395
    font = cv2.FONT_HERSHEY_SIMPLEX
    cy_choice = 854

    def draw_bracket1(img): draw_bracket(img, 231, 21, y_bracket, "21")
    def draw_bracket2(img): draw_bracket(img, 402, 39, y_bracket, "39")
    def draw_plus1(img): cv2.putText(img, "+18", (315, 395), font, 0.7, (0, 120, 0), 2)
    def draw_bracket3(img): draw_bracket(img, 574, 57, y_bracket, "57")
    def draw_plus2(img): cv2.putText(img, "+18", (495, 395), font, 0.7, (0, 120, 0), 2)
    def draw_plus3(img): cv2.putText(img, "+18", (675, 395), font, 0.7, (0, 120, 0), 2)
    def draw_bracket4(img): draw_bracket(img, 741, 83, y_bracket, "75")
    
    def draw_cross1(img): draw_cross(img, 61 + 105, cy_choice)
    def draw_cross2(img): draw_cross(img, 521 + 105, cy_choice)
    def draw_cross3(img): draw_cross(img, 751 + 105, cy_choice)

    elements = [
        (2, 6, draw_bracket1),
        (6, 10, draw_bracket2),
        (10, 14, draw_plus1),
        (14, 18, draw_bracket3),
        (18, 22, draw_plus2),
        (22, 26, draw_plus3),
        (26, 30, draw_bracket4),
        (31, 35, draw_cross1),
        (35, 39, draw_cross2),
        (39, 43, draw_cross3)
    ]
    circle_start = 43
    circle_end = 51

    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', pixelformat='yuv420p', macro_block_size=None)

    for f in range(60):
        img = original_img.copy()
        for start, end, func in elements:
            if f < start:
                alpha = 0
            elif f >= end:
                alpha = 1
            else:
                alpha = (f - start) / (end - start)
            img = draw_alpha(img, func, alpha)

        if f > circle_start:
            if f >= circle_end:
                progress = 1.0
            else:
                progress = (f - circle_start) / (circle_end - circle_start)
            cv2.ellipse(img, (291 + 105, cy_choice), (100, 100), 0, -90, int(-90 + 360 * progress), (0, 0, 255), 4)

        # Convert from BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        writer.append_data(img_rgb)

    writer.close()

if __name__ == '__main__':
    main()
