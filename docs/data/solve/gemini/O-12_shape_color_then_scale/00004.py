import cv2
import numpy as np
import imageio
import os

def solve():
    img = cv2.imread('/app/first_frame.png')
    
    # Pre-extract elements
    clean = img.copy()
    # Erase QM1 and QM2
    clean[655:710, 500:540] = 255
    clean[655:710, 836:876] = 255
    
    QM1 = img[655:710, 500:540].copy()
    QM2 = img[655:710, 836:876].copy()
    
    D = img[633:633+99, 82:82+197].copy()
    mask_D = (D != [255, 255, 255]).any(axis=-1).astype(np.uint8)
    
    color_106 = np.array([30, 132, 153], dtype=np.float32)
    color_236 = np.array([149, 191, 66], dtype=np.float32)
    
    def draw_QM(frame, qm, x, y, alpha):
        if alpha <= 0: return
        roi = frame[y:y+qm.shape[0], x:x+qm.shape[1]].astype(np.float32)
        qm_f = qm.astype(np.float32)
        roi = roi * (1 - alpha) + qm_f * alpha
        frame[y:y+qm.shape[0], x:x+qm.shape[1]] = roi.astype(np.uint8)

    def draw_shape_rgb(frame, shape, cx, cy):
        h, w = shape.shape[:2]
        x0 = int(round(cx - w / 2))
        y0 = int(round(cy - h / 2))
        roi = frame[y0:y0+h, x0:x0+w]
        mask = (shape != [255, 255, 255]).any(axis=-1)
        # To avoid issues with out-of-bounds
        if y0 < 0 or y0+h > frame.shape[0] or x0 < 0 or x0+w > frame.shape[1]:
            # crop shape and roi
            y1_f = max(0, y0)
            y2_f = min(frame.shape[0], y0+h)
            x1_f = max(0, x0)
            x2_f = min(frame.shape[1], x0+w)
            
            y1_s = y1_f - y0
            y2_s = y2_f - y0
            x1_s = x1_f - x0
            x2_s = x2_f - x0
            
            roi_cropped = frame[y1_f:y2_f, x1_f:x2_f]
            shape_cropped = shape[y1_s:y2_s, x1_s:x2_s]
            mask_cropped = mask[y1_s:y2_s, x1_s:x2_s]
            
            roi_cropped[mask_cropped] = shape_cropped[mask_cropped]
            frame[y1_f:y2_f, x1_f:x2_f] = roi_cropped
        else:
            roi[mask] = shape[mask]
            frame[y0:y0+h, x0:x0+w] = roi

    frames = []
    
    for f in range(60):
        frame = clean.copy()
        
        if f < 30:
            progress = f / 29.0
            a_qm1 = 1.0 - progress
            a_qm2 = 1.0
            
            draw_QM(frame, QM1, 500, 655, a_qm1)
            draw_QM(frame, QM2, 836, 655, a_qm2)
            
            cx = 180.5 + (518.5 - 180.5) * progress
            cy = 682.5 + (682.0 - 682.5) * progress
            current_color = color_106 * (1 - progress) + color_236 * progress
            
            shape = D.copy()
            shape[np.all(D == [30, 132, 153], axis=-1)] = current_color
            
            draw_shape_rgb(frame, shape, cx, cy)
            
        else:
            progress = (f - 30) / 29.0
            a_qm1 = 0.0
            a_qm2 = 1.0 - progress
            
            draw_QM(frame, QM2, 836, 655, a_qm2)
            
            # Static E
            shape_E = D.copy()
            shape_E[np.all(D == [30, 132, 153], axis=-1)] = color_236
            draw_shape_rgb(frame, shape_E, 518.5, 682.0)
            
            # Moving object 2
            cx = 518.5 + (854.5 - 518.5) * progress
            cy = 682.0
            scale = 1.0 + (0.5 - 1.0) * progress
            
            w = max(1, int(round(197 * scale)))
            h = max(1, int(round(99 * scale)))
            
            mask_scaled = cv2.resize(mask_D, (w, h), interpolation=cv2.INTER_NEAREST)
            mask_padded = np.pad(mask_scaled, 10, mode='constant', constant_values=0)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            interior_padded = cv2.erode(mask_padded, kernel)
            interior = interior_padded[10:-10, 10:-10]
            
            shape = np.full((h, w, 3), 255, dtype=np.uint8)
            shape[mask_scaled == 1] = [0, 0, 0]
            shape[interior == 1] = color_236
            
            draw_shape_rgb(frame, shape, cx, cy)
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    os.makedirs('/app/output', exist_ok=True)
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', pixelformat='yuv420p')
    for frm in frames:
        writer.append_data(frm)
    writer.close()

if __name__ == '__main__':
    solve()
