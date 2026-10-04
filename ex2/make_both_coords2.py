import csv

path_world = r'C:\Users\User\Desktop\блендер\true_coords_world.csv'
path_pixels = r'C:\Users\User\Desktop\блендер\true_coords_pixels.csv'

# Параметры
fps = 25
duration = 10
total_frames = fps * duration  

# Камеры
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
    """Проекция мировой точки на изображение камеры."""
    x_cam = x
    y_cam = y - cam_y
    z_cam = z
    u = cx + f_px * (y_cam / x_cam)
    v = cy - f_px * (z_cam / x_cam)
    return u, v

#Файл 1: Мировые координаты
with open(path_world, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f, delimiter=';')
    writer.writerow(['frame', 'time',
                     'Xr', 'Yr', 'Zr',
                     'Xl', 'Yl', 'Zl'])
    
    for frame in range(0, total_frames + 1):
        t = frame / fps
        
        # RightBall
        xr = 300 - 20 * t
        yr = -100 + 20 * t
        zr = 0.0
        
        # LeftBall
        xl = 300 - 10 * t
        yl = 100 - 20 * t
        zl = 0.0
        
        writer.writerow([frame, round(t, 4),
                         round(xr, 4), round(yr, 4), round(zr, 4),
                         round(xl, 4), round(yl, 4), round(zl, 4)])

#Файл 2
with open(path_pixels, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f, delimiter=';')
    writer.writerow([
        'frame', 'time',
        'ur_L', 'vr_L',   # RightBall в LeftCamera
        'ur_R', 'vr_R',   # RightBall в RightCamera
        'ul_L', 'vl_L',   # LeftBall в LeftCamera
        'ul_R', 'vl_R'    # LeftBall в RightCamera
    ])
    
    for frame in range(0, total_frames + 1):
        t = frame / fps
        
        # RightBall
        xr = 300 - 20 * t
        yr = -100 + 20 * t
        zr = 0.0
        
        # LeftBall
        xl = 300 - 10 * t
        yl = 100 - 20 * t
        zl = 0.0
        
        
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

pr