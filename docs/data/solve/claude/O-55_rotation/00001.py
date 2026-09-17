#!/usr/bin/env python3
"""Regenerate /app/output/video.mp4: orthographic camera orbit (az 10 -> 190 deg,
elev 36 deg) around a fixed 6-cube sculpture, matching /app/first_frame.png."""
import math, os, subprocess, sys, tempfile
import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")

W = H = 1024
FPS = 16
N_FRAMES = 21
SS = 1  # the reference is rendered without anti-aliasing (5 exact colors, 1px lines)

# Voxel positions (x right, y depth/back, z up), unit cubes with min corner here.
CUBES = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 1, 1), (0, 1, 2), (0, 0, 2)]

BG = (255, 255, 255)
COL_TOP = (197, 168, 225)
COL_FRONT = (157, 135, 180)   # faces normal to y axis
COL_SIDE = (118, 101, 135)    # faces normal to x axis
EDGE = (0, 0, 0)

# Camera parameters (fitted to first_frame.png; see fit_camera()).
AZ0, AZ1 = 10.0, 190.0
ELEV = 36.0
PARAMS = dict(scale=137.9, cx=512.2, cy=512.5, pivot=(1.0, 1.0, 1.5))  # 22/1M px mismatch vs first frame

# Face definitions: (normal, 4 corner offsets in CCW order seen from outside)
FACES = {
    (0, 0, 1): [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)],
    (0, 0, -1): [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)],
    (0, -1, 0): [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)],
    (0, 1, 0): [(1, 1, 0), (0, 1, 0), (0, 1, 1), (1, 1, 1)],
    (1, 0, 0): [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)],
    (-1, 0, 0): [(0, 1, 0), (0, 0, 0), (0, 0, 1), (0, 1, 1)],
}


def camera(az_deg, el_deg):
    az, el = math.radians(az_deg), math.radians(el_deg)
    cam_dir = np.array([math.sin(az) * math.cos(el), -math.cos(az) * math.cos(el), math.sin(el)])
    right = np.array([math.cos(az), math.sin(az), 0.0])
    up = np.array([-math.sin(az) * math.sin(el), math.cos(az) * math.sin(el), math.cos(el)])
    return cam_dir, right, up


def face_color(normal, cam_dir):
    """Fixed world shading matching the source render: base colour x 1.0 (top),
    x 0.8 (faces normal to y) and x 0.6 (faces normal to x). Independent of the
    camera so the sculpture's appearance is continuous while the camera orbits."""
    nx, ny, nz = normal
    if nz > 0:
        return COL_TOP
    if nz < 0:
        return COL_SIDE
    return COL_FRONT if ny != 0 else COL_SIDE


def project(pts, cam_dir, right, up, p):
    pts = np.asarray(pts, float) - np.asarray(p["pivot"], float)
    u = p["cx"] + p["scale"] * (pts @ right)
    v = p["cy"] - p["scale"] * (pts @ up)
    d = pts @ cam_dir
    return u, v, d


def render(az_deg, el_deg, p, ss=SS, line_w=None):
    cam_dir, right, up = camera(az_deg, el_deg)
    Wss, Hss = W * ss, H * ss
    img = Image.new("RGB", (Wss, Hss), BG)
    draw = ImageDraw.Draw(img)
    if line_w is None:
        line_w = ss
    # painter's algorithm: sort cubes far -> near by center depth along cam_dir
    order = sorted(CUBES, key=lambda c: np.dot(np.array(c) + 0.5, cam_dir))
    for c in order:
        faces = []
        for n, corners in FACES.items():
            if np.dot(n, cam_dir) <= 1e-9:
                continue  # back-face cull
            pts = [(c[0] + o[0], c[1] + o[1], c[2] + o[2]) for o in corners]
            u, v, d = project(pts, cam_dir, right, up, p)
            faces.append((d.mean(), n, list(zip((u * ss).tolist(), (v * ss).tolist()))))
        faces.sort(key=lambda f: f[0])
        for _, n, poly in faces:
            draw.polygon(poly, fill=face_color(n, cam_dir))
        for _, n, poly in faces:
            draw.line(poly + [poly[0]], fill=EDGE, width=line_w, joint="curve")
    if ss > 1:
        img = img.resize((W, H), Image.BOX)
    return img


def fit_camera(ref):
    """Refine scale/cx/cy (and check pivot) against the reference frame by pixel MSE."""
    from scipy.optimize import minimize
    ref = np.asarray(ref, float)

    def loss(x):
        p = dict(PARAMS, scale=x[0], cx=x[1], cy=x[2])
        im = np.asarray(render(AZ0, ELEV, p, ss=2), float)
        return np.mean((im - ref) ** 2)

    x0 = [PARAMS["scale"], PARAMS["cx"], PARAMS["cy"]]
    res = minimize(loss, x0, method="Nelder-Mead", options=dict(xatol=0.05, fatol=0.01, maxiter=300))
    return res


def ease(t):
    return t  # linear pacing (constant angular speed)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ref = Image.open(FIRST).convert("RGB")
    frames_dir = tempfile.mkdtemp(prefix="frames_")
    for i in range(N_FRAMES):
        t = i / (N_FRAMES - 1)
        az = AZ0 + (AZ1 - AZ0) * ease(t)
        if i == 0:
            img = ref  # first frame is the given image exactly
        else:
            img = render(az, ELEV, PARAMS)
        img.save(os.path.join(frames_dir, f"f{i:03d}.png"))
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
        "-i", os.path.join(frames_dir, "f%03d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
        "-r", str(FPS), OUT,
    ], check=True)
    print("wrote", OUT)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "fit":
        ref = Image.open(FIRST).convert("RGB")
        print(fit_camera(ref))
    else:
        main()
