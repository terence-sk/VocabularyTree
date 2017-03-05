import cv2
import numpy as np

img = cv2.imread('test_image.jpg', cv2.IMREAD_GRAYSCALE)

# channels for color image
height, width = img.shape

# blank image, will contain image and detected keypoints, just to show it works
blank_image = np.zeros((height, width, 3), np.float32)


mser = cv2.MSER_create()
# detecting keypoints
mser_areas = mser.detect(img)

#creating sift to compute descriptors
sift = cv2.xfeatures2d.SIFT_create()

# descriptors created from MSER keypoints teda aspon dufam
descs = sift.compute(img, mser_areas)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

#TypeError: data is not a numerical tuple -- v tomto pripade data = descs
# AKO MAM SPRAVIT Z KEYPOINTOV TUPLE?
ret,label,center = cv2.kmeans(data = descs[1], K = 6, bestLabels=None, criteria=criteria, attempts=10, flags=cv2.KMEANS_RANDOM_CENTERS)

print("ret=",ret,"label=",label,"center=",center)

#error: (-215) data0.dims <= 2 && type == CV_32F && K > 0 in function kmeans
#ret,label,center = cv2.kmeans((100,200),6 ,None, criteria,10,cv2.KMEANS_RANDOM_CENTERS)