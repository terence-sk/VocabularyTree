import os
import cv2
from TreeObject import TreeObject
from KeyValuePair import KeyValuePair
from Pic import Pic
import numpy as np
import math

MINIMUM_KEYPOINTS = 10
NUM_OF_CLUSTERS = 10
NUM_OF_LEVELS = 3
NUM_OF_PICS = -1


def DEBUG_load_two():
    current_dir = os.getcwd()
    pics = current_dir + "/pics"

    pic_list = []
    img = cv2.imread(pics + "/" + "house1.jpg", cv2.IMREAD_GRAYSCALE)
    pic_list.append(Pic("house1.jpg", img))

    img = cv2.imread(pics + "/" + "house2.jpg", cv2.IMREAD_GRAYSCALE)
    pic_list.append(Pic("house2.jpg", img))

    global NUM_OF_PICS
    NUM_OF_PICS = len(pic_list)

    return pic_list


def load_pics():

    current_dir = os.getcwd()
    pics = current_dir + "/pics"

    pic_list = []

    for filename in os.listdir(pics):
        img = cv2.imread(pics + "/" + filename, cv2.IMREAD_GRAYSCALE)
        pic_list.append(Pic(filename, img))

    global NUM_OF_PICS
    NUM_OF_PICS = len(pic_list)

    return pic_list


def get_keypoint_descriptors_tuple(img):
    # creating MSER keypoint detector
    mser = cv2.MSER_create()

    # detecting keypoints
    mser_areas = mser.detect(img)

    # creating SIFT object to compute descriptors
    sift = cv2.xfeatures2d.SIFT_create()

    # descriptors created from MSER keypoints
    keypoints, descriptors = sift.compute(img, mser_areas)

    return keypoints, descriptors


def get_clustered_data(descs):
    #  10 = Flag to specify the number of times the algorithm is executed using different initial labellings.
    # The algorithm returns the labels that yield the best compactness.
    # This compactness is returned as output. (?)
    # 1.0 = accuracy
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    # Kolko tych clustrov vlastne ma byt..?
    # -- run k means up to 6 levels - pustim k means, mam x clustrov, nad tymi x clustrami zase k means
    # a takto to spravim max 6 krat
    # TODO: branch factor je odporucany na 10 ale to este neviem ako ovplyvnim

    # center = list of 6 arrays of length 128 - descriptors of cluster centers
    # compactness = It is the sum of squared distance from each point to their corresponding centers.
    # label = Label of keypoint to which cluster does keypoint belong
    compactness, labels, centers = cv2.kmeans(data=descs, K=NUM_OF_CLUSTERS, bestLabels=None, criteria=criteria, attempts=10,
                                      flags=cv2.KMEANS_RANDOM_CENTERS)

    return compactness, labels, centers


def create_tree(item):

    # Not sure about this one but I'll just let it be here for a while
    if len(item.get_kvps().get_value()) <= 10:
        return

    item.setup(NUM_OF_PICS)

    compactness, labels, center = get_clustered_data(item.get_kvps().get_value())

    descriptors = item.get_kvps().get_value()
    paths = item.get_kvps().get_key()

    for i in range(0, NUM_OF_CLUSTERS):

        desc = descriptors[labels.ravel() == i]
        path = paths[labels.ravel() == i]
        kvp = KeyValuePair(path, desc)

        item.add_child(TreeObject(kvp, compactness=compactness, label=i, center=center))

        create_tree(item.get_child(i))

    # print(len(item.getkvps().getkey()))


def main():
    # Toto je moj root (Descriptors with paths to corresponding images)
    tree = []
    descriptors = []
    paths = []
    pics = DEBUG_load_two()

    for pic in pics:
        kp, desc = get_keypoint_descriptors_tuple(pic.img)

        if kp is None or desc is None:
            print("No descriptors for " + pic.path)
            continue

        descriptors.append(desc)

        # aby som vedel ktory deskriptor patri ku ktoremu obrazku
        for i in range(0, len(desc)):
            paths.append(pic.path)

    # spoji list x poli (obrazkov) pricom kazde pole obsahuje
    # samo o sebe y descriptorov
    descriptors = np.vstack(descriptors)
    paths = np.asarray(paths)

    compactness, labels, center = get_clustered_data(descriptors)

    # initial list from which the tree is created
    for i in range(0, NUM_OF_CLUSTERS):
        desc = descriptors[labels.ravel() == i]
        path = paths[labels.ravel() == i]
        kvp = KeyValuePair(path, desc)

        tree.append(TreeObject(kvp, compactness=compactness, label=i, center=center))

    for node in tree:
        create_tree(node)

    print('main ends')


# starting point
if __name__ == '__main__':
    main()
    print("DONE")
