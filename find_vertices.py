import cv2
import numpy as np
from utils import read_video

frame = read_video(r'input_videos\A1606b0e6_0 (10).mp4')[0]

pixel_vertices = np.array([
   [300,  62],    # top-left
        [1900, 95],    # top-right
        [1700, 970],  # bottom-right
        [8,    956]   
], dtype=np.int32)

cv2.polylines(frame, [pixel_vertices], isClosed=True, color=(0,255,0), thickness=3)
for point in pixel_vertices:
    cv2.circle(frame, tuple(point), 8, (0,0,255), -1)

cv2.imwrite('output_videos/vertices_visualization.jpg', frame)
print("Saved")