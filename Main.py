import os
import cv2
from TreeObject import TreeObject
from KeyValuePair import KeyValuePair
from Pic import Pic
import numpy as np


def loadPics():

    currentDir = os.getcwd()
    picDir = currentDir + "/pics"

    pics = []

    for filename in os.listdir(picDir):
        img = cv2.imread(picDir + "/" + filename, cv2.IMREAD_GRAYSCALE)
        pics.append(Pic(filename, img))

    return pics


def getKeypointDescriptorsTuple(img):
    # creating MSER keypoint detector
    mser = cv2.MSER_create()

    # detecting keypoints
    mser_areas = mser.detect(img)

    # creating SIFT object to compute descriptors
    sift = cv2.xfeatures2d.SIFT_create()

    # descriptors created from MSER keypoints
    keypoints, descriptors = sift.compute(img, mser_areas)

    return keypoints, descriptors


NUM_OF_CLUSTERS = 10
NUM_OF_LEVELS = 3


def getClusteredData(descriptors):
    #  10 = Flag to specify the number of times the algorithm is executed using different initial labellings. The algorithm returns the labels that yield the best compactness.
    # This compactness is returned as output. (?)
    # 1.0 = accuracy
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    # Kolko tych clustrov vlastne ma byt..?
    # -- run k means up to 6 levels - pustim k means, mam x clustrov, nad tymi x clustrami zase k means
    # a takto to spravim 6 krat
    # TODO: branch factor je odporucany na 10 ale to este neviem ako ovplyvnim

    # center = list of 6 arrays of length 128 - descriptors of cluster centers
    # ret = It is the sum of squared distance from each point to their corresponding centers.
    # label = Label of keypoint to which cluster does keypoint belong
    ret, label, center = cv2.kmeans(data=descriptors, K=NUM_OF_CLUSTERS, bestLabels=None, criteria=criteria,
                                    attempts=10,
                                    flags=cv2.KMEANS_RANDOM_CENTERS)

    return ret, label, center


# starting point
if __name__ == '__main__':


    # Toto je moj root (Descriptors with paths to corresponding images)
    tree = []
    descriptors = []
    paths = []
    pics = loadPics()

    for pic in pics:
        kp, desc = getKeypointDescriptorsTuple(pic.img)
        descriptors.append(desc)

        # aby som vedel ktory deskriptor patri ku ktoremu obrazku
        for i in range(0, len(desc)):
            paths.append(pic.path)


    # spoji list 5tich poli (5 obrazkov po X*128)
    descriptors = np.vstack(descriptors)
    paths = np.asarray(paths)

    ret, labels, center = getClusteredData(descriptors)

    for i in range(0, NUM_OF_CLUSTERS):
        desc = descriptors[labels.ravel() == i]
        path = paths[labels.ravel() == i]
        kvp = KeyValuePair(path, desc)

        tree.append(TreeObject(kvp, i))


    print("breakpoint")