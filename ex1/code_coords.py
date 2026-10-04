import csv

path = r'C:\Users\User\Desktop\блендер видео\coords.csv'

start_x = 300.0      
end_x = 100.0        
fps = 25
duration = 10       
total_frames = fps * duration 

with open(path, 'w', newline='') as f:
    writer = csv.writer(f)

    writer.writerow(['X', 'Y', 'Z'])

    for frame in range(0, total_frames + 1):
        x = start_x - (start_x - end_x) * (frame / total_frames)
        y = 0.0
        z = 0.0
        writer.writerow([x, y, z])
