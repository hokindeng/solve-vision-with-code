import cv2
import numpy as np
import imageio
import os

def intersects(gx, gy, rects, bw=42, bh=42):
    bx, by = gx - 21, gy - 21
    for rx, ry, rw, rh in rects:
        if not (bx + bw < rx or bx > rx + rw or by + bh < ry or by > ry + rh):
            return True
    return False

def main():
    img = cv2.imread('/app/first_frame.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=20, minRadius=10, maxRadius=50)
    circles = np.uint16(np.around(circles))[0, :]

    clusters = {}
    for c in circles:
        x, y, r = int(c[0]), int(c[1]), int(c[2])
        color = tuple(img[y, x].tolist())
        if color in [(255, 255, 255), (240, 240, 240), (229, 229, 229)]:
            continue
        if color not in clusters: clusters[color] = []
        clusters[color].append((x, y))

    # sort clusters points for determinism
    for k in clusters:
        clusters[k].sort(key=lambda p: (p[0], p[1]))

    pts_A = clusters.get((50, 50, 255), [])

    rects = [
        (385, 275, 114, 43),
        (239, 494, 150, 43),
        (453, 746, 113, 43),
        (603, 489, 112, 44),
        (5, 14, 322, 48)
    ]

    targets = []
    for c in range(-6, 7):
        for r in range(-6, 7):
            gx = 315 + c * 46
            gy = 279 + r * 46
            valid = True
            for px, py in pts_A + targets:
                if (gx - px)**2 + (gy - py)**2 < 42**2:
                    valid = False
                    break
            if valid and not intersects(gx, gy, rects) and 50 < gx < 974 and 50 < gy < 974:
                targets.append((gx, gy))

    targets.sort(key=lambda p: (p[0]-315)**2 + (p[1]-279)**2)

    balls = []
    clean_bg = img.copy()

    # D: Yellow, C: Turquoise, B: Purple
    color_to_frames = {
        (50, 50, 255): (0, 0),      # Red
        (50, 220, 255): (10, 25),   # Yellow
        (208, 224, 64): (35, 50),   # Turquoise
        (255, 50, 200): (60, 75)    # Purple
    }

    t_idx = 0
    for color, pts in clusters.items():
        for x, y in pts:
            sprite = img[y-24:y+24, x-24:x+24].copy()
            
            mask = np.any(sprite != [240, 240, 240], axis=-1)
            clean_bg[y-24:y+24, x-24:x+24][mask] = [240, 240, 240]

            sf, ef = color_to_frames[color]
            
            if color == (50, 50, 255):
                tx, ty = x, y
            else:
                tx, ty = targets[t_idx]
                t_idx += 1
                
            balls.append({
                'start_x': x, 'start_y': y,
                'target_x': tx, 'target_y': ty,
                'start_frame': sf, 'end_frame': ef,
                'color': color,
                'sprite': sprite
            })
            
    # sort balls so red is drawn first
    balls.sort(key=lambda b: 0 if b['color'] == (50, 50, 255) else 1)

    os.makedirs('/app/output', exist_ok=True)
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', pixelformat='yuv420p', macro_block_size=None)

    for f in range(80):
        frame = clean_bg.copy()
        for b in balls:
            sx, sy = b['start_x'], b['start_y']
            tx, ty = b['target_x'], b['target_y']
            sf, ef = b['start_frame'], b['end_frame']
            color = b['color']
            sprite = b['sprite']

            if sf == ef:
                progress = 1.0
            else:
                if f < sf:
                    progress = 0.0
                elif f > ef:
                    progress = 1.0
                else:
                    p = (f - sf) / (ef - sf)
                    progress = p * p * (3 - 2 * p)

            cx = int(sx + (tx - sx) * progress)
            cy = int(sy + (ty - sy) * progress)

            sprite_to_draw = sprite.copy()
            if progress == 1.0 and color != (50, 50, 255):
                mask_color = np.all(sprite_to_draw == color, axis=-1)
                sprite_to_draw[mask_color] = (50, 50, 255)

            mask = np.any(sprite_to_draw != [240, 240, 240], axis=-1)
            
            ry1, ry2 = cy - 24, cy + 24
            rx1, rx2 = cx - 24, cx + 24
            
            # just in case it goes out of bounds, though it shouldn't
            if ry1 >= 0 and rx1 >= 0 and ry2 <= 1024 and rx2 <= 1024:
                frame[ry1:ry2, rx1:rx2][mask] = sprite_to_draw[mask]
            else:
                print(f"Warning: Ball out of bounds at {cx}, {cy}")
        
        # imageio expects RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_rgb)
    
    writer.close()
    print("Done")

if __name__ == '__main__':
    main()
