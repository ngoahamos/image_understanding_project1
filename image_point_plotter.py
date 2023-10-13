import os;
from load_ground_truth import  ground_loader
import cv2
import numpy as np

image = cv2.imread('test_image.jpg')

harris_points = ground_loader('harris_points.txt')
# log_points = ground_loader('log_points.txt')
sift_points = ground_loader('ground_truth.txt')

harris_color = (0,0,255)
log_color = (255,0,0)
sift_color = (0,0,0)

for point in sift_points:
    x,y = map(int, point)
    cv2.circle(image, (x,y), 5, sift_color, -1)

# for point in log_points:
#     x, y = map(int, point)
#     cv2.circle(image, (x, y), 5, log_color, -1)

for point in harris_points:
    x, y = map(int, point)
    cv2.circle(image, (x, y), 5, harris_color, -1)

cv2.imshow("Image Points", image)
cv2.waitKey(0)
cv2.destroyAllWindows()