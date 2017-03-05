import cv2
import numpy as np

img = cv2.imread('test_image.jpg', cv2.IMREAD_GRAYSCALE)

# channels for color image, if needed, but I was told to use greyscale
height, width = img.shape

# blank image, will contain image and detected keypoints, just to show it works
# blank_image = np.zeros((height, width, 3), np.float32)

# creating MSER keypoint detector
mser = cv2.MSER_create()
# detecting keypoints
mser_areas = mser.detect(img)

# creating SIFT object to compute descriptors
sift = cv2.xfeatures2d.SIFT_create()

# descriptors created from MSER keypoints (IMHO :D)
descs = sift.compute(img, mser_areas)

#  10 = Flag to specify the number of times the algorithm is executed using different initial labellings. The algorithm returns the labels that yield the best compactness. This compactness is returned as output. (?)
# 1.0 = accuracy (TODO: Ask accuracy of WHAT?)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

ret,label,center = cv2.kmeans(data = descs[1], K = 6, bestLabels=None, criteria=criteria, attempts=10, flags=cv2.KMEANS_RANDOM_CENTERS)

