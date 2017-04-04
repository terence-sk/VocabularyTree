import os
import cv2
from TreeObject import TreeObject
from TreeTypeEnum import TreeTypeEnum
from KeyValuePair import KeyValuePair
from Pic import Pic
import numpy as np
import sys

MINIMUM_KEYPOINTS = 10
NUM_OF_CLUSTERS = 10
NUM_OF_LEVELS = 3
NUM_OF_PICS = -1
TREE_TYPE = None


def load_pics():

    current_dir = os.getcwd()

    #pics = current_dir + "/pics" TODO Revert
    pics = current_dir + "/DEBUG_test_images_few"

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


def tree_add_node(item):

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

        tree_add_node(item.get_child(i))

    # item.to_string()


def create_tree():
    # Toto je moj root (Descriptors with paths to corresponding images)
    tree = []
    descriptors = []
    paths = []

    pics = load_pics()

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
        tree_add_node(node)

    print('main ends')
    return tree


def get_query_image_descriptors():
    query_path = os.getcwd() + "/query_pic/query_tux_iny_tux.jpg"
    query_img = cv2.imread(query_path, cv2.IMREAD_GRAYSCALE)
    kp, desc = get_keypoint_descriptors_tuple(query_img)

    return desc


def get_closest_node(database, descriptor):

    for node in database:
        # najmensia vzdialenost medzi centrom a deskriptorom
        lowest = sys.float_info.max
        index = -1

        # uzly na rovnakej urovni stromu mam v list-e, takisto aj ich centra
        center_list = node.get_center_list()

        # porovnavam jednotlive centra s deskriptorom
        # a hladam take ktoremu je najblizsi
        for idx, center in enumerate(center_list):
            # euclidean distance
            dist = abs(np.linalg.norm(center - descriptor))
            if dist < lowest:
                lowest = dist
                index = idx
                database[index].visit()

        # prechadzame uzly pokial maju deti
        if len(database[index].get_child_all()) != 0:
            database = database[index].get_child_all()
        # dosiahli sme leaf node
        # uzol uz deti nema, ale moze mat viac ako jeden descriptor
        else:
            lowest = sys.float_info.max
            most_similar_object = None

            # ak ostal uz len jeden KVP alebo su vsetky KVP z jedneho obrazku, mozme rovno vratit jeho nazov
            if len(database[index].get_kvps().get_key()) == 1 or len(np.unique(database[index].get_kvps().get_key())) == 1:
                return database[index].get_kvps().get_key()[0]

            # v opacnom pripade prejdeme este deskriptory leaf nodu
            else:
                leaf_node_descs = database[index].get_kvps()

                for idx, val in enumerate(leaf_node_descs.get_value()):
                    dist = abs(np.linalg.norm(val - descriptor))

                    if dist < lowest:
                        lowest = dist
                        most_similar_object = leaf_node_descs.get_key()[idx]

            return most_similar_object


# starting point
if __name__ == '__main__':

    global TREE_TYPE

    TREE_TYPE = TreeTypeEnum.CREATE
    database = create_tree()

    descriptors = get_query_image_descriptors()
    print("pocet najdenych deskriptorov pre query image: ", len(descriptors))
    ## THIS IS JUST DEBUG TODO REMOVE
    #print(get_closest_node(database, descriptors[0]))

    #for desc in descriptors:
    #    print(get_closest_node(database, desc))

    print("DONE")
