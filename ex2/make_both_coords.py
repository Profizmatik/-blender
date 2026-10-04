import csv
import math

path_world = r'C:\Users\User\Desktop\блендер\true_coords_world2.csv'
path_pixels = r'C:\Users\User\Desktop\блендер\true_coords_pixels2.csv'

fps = 25
duration = 10
total_frames = fps * duration  # 250

# Параметры камер
f_mm = 50.0
W_sensor = 36.0
W_px = 768
H_px = 576
f_px = f_mm * W_px / W_sensor
cx = W_px / 2
cy = H_px / 2

cam_left_y = 15.0
cam_right_y = -15.0

def project(x, y, z, cam_y):
    x_cam = x
    y_cam = y - cam_y
    z_cam = z
    u = cx + f_px * (y_cam / x_cam)
    v = cy - f_px * (z_cam / x_cam)
    return u, v

# ============ Файл 1: Мировые ============
with open(path_world, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f, delimiter=';')
    writer.writerow(['frame', 'time',
                     'Xr', 'Yr', 'Zr',
                     'Xl', 'Yl', 'Zl'])
    
    for frame in range(0, total_frames + 1):
        t = frame / fps
        
        # RightBall: (300, -100) -> (100, 50), Z = 0
        xr = 300 - 20 * t
        yr = -100 + 15 * t
        zr = 0.0
        
        # LeftBall: X=300, Y от 100 до -100, Z — синусоида (амплитуда 10)
        xl = 300.0
        yl = 100 - 20 * t
        zl = 10 * math.sin(2 * math.pi * 0.5 * t)
        
        writer.writerow([frame, round(t, 4),
                         round(xr, 4), round(yr, 4), round(zr, 4),
                         round(xl, 4), round(yl, 4), round(zl, 4)])

# ============ Файл 2: Пиксельные ============
with open(path_pixels, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f, delimiter=';')
    writer.writerow([
        'frame', 'time',
        'ur_L', 'vr_L', 'ur_R', 'vr_R',
        'ul_L', 'vl_L', 'ul_R', 'vl_R'
    ])
    
    for frame in range(0, total_frames + 1):
        t = frame / fps
        
        # RightBall
        xr = 300 - 20 * t
        yr = -100 + 15 * t
        zr = 0.0
        
        # LeftBall
        xl = 300.0
        yl = 100 - 20 * t
        zl = 10 * math.sin(2 * math.pi * 0.5 * t)
        
        ur_L, vr_L = project(xr, yr, zr, cam_left_y)
        ur_R, vr_R = project(xr, yr, zr, cam_right_y)
        ul_L, vl_L = project(xl, yl, zl, cam_left_y)
        ul_R, vl_R = project(xl, yl, zl, cam_right_y)
        
        writer.writerow([
            frame, round(t, 4),
            round(ur_L, 2), round(vr_L, 2),
            round(ur_R, 2), round(vr_R, 2),
            round(ul_L, 2), round(vl_L, 2),
            round(ul_R, 2), round(vl_R, 2)
        ])

print("Готово!")
print("Мировые:", path_world)
print("Пиксельные:", path_pixels)