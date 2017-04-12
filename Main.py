import os
import cv2
from TreeObject import TreeObject
from KeyValuePair import KeyValuePair
from Pic import Pic
import numpy as np
import sys
from copy import deepcopy
from collections import Counter
import time

MINIMUM_KEYPOINTS = 10
NUM_OF_CLUSTERS = 10
NUM_OF_LEVELS = 3
NUM_OF_PICS = -1
QUERY_TREE = []

DB_SIZE = 0
DB_UNIQUE_ITEMS = []


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


def tree_add_node(item):

    item.setup(NUM_OF_PICS)

    # Not sure about this one but I'll just let it be here for a while
    if len(item.get_kvps().get_value()) <= 10:
        return

    compactness, labels, center = get_clustered_data(item.get_kvps().get_value())

    descriptors = item.get_kvps().get_value()
    paths = item.get_kvps().get_key()

    for i in range(0, NUM_OF_CLUSTERS):

        desc = descriptors[labels.ravel() == i]
        path = paths[labels.ravel() == i]
        kvp = KeyValuePair(path, desc)

        # TODO: zistit ci pridavanie parenta nezvysi pamatovu narocnost prilis, lebo toto je velmi jednoduche riesenie
        item.add_child(TreeObject(kvp, compactness=compactness, label=i, center=center[i]), parent=item)

        tree_add_node(item.get_child(i))


def create_tree():

    # Toto je moj root (Descriptors with paths to corresponding images)
    tree = []
    descriptors = []
    paths = []

    start = time.time()

    pics = load_pics()

    print('Pics loaded in ' , time.time() - start , 's')

    start = time.time()

    for pic in pics:
        kp, desc = get_keypoint_descriptors_tuple(pic.img)

        if kp is None or desc is None:
            print("No descriptors for " + pic.path)
            continue

        descriptors.append(desc)

        # aby som vedel ktory deskriptor patri ku ktoremu obrazku
        for i in range(0, len(desc)):
            paths.append(pic.path)

    print('Descriptor retrieval from ',NUM_OF_PICS,' images took ', time.time() - start, 's')

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

        tree.append(TreeObject(kvp, compactness=compactness, label=i, center=center[i]))

    for node in tree:
        tree_add_node(node)

    return tree


def get_query_image_descriptors():
    query_path = os.getcwd() + "/query_pic/query_tux_iny_tux.jpg"
    query_img = cv2.imread(query_path, cv2.IMREAD_GRAYSCALE)
    kp, desc = get_keypoint_descriptors_tuple(query_img)

    return desc


def create_query_tree(item_list):

    if len(item_list) == 0:
        return

    for idx, item in enumerate(item_list):

        if item.get_num_of_visits() == 0:
            item_list[idx] = None
            continue
        item.query_setup()
        create_query_tree(item.get_child_all())


# len pooznacuje visits v databazovych objektoch
def mark_node_visits_in_db(database, descriptor):
    most_similar_object = None
    while True:

        lowest = sys.float_info.max
        index = -1

        # list 10tich TreeObjectov
        for idx, node in enumerate(database):
            # porovnavam jednotlive centra s deskriptorom
            # a hladam take ktoremu je najblizsi
            dist = abs(np.linalg.norm(node.get_center() - descriptor))
            if dist < lowest:
                lowest = dist
                index = idx
                #print("navstivil ", str(index))
        #print("lowest ", str(index))
        database[index].visit()

        # prechadzame uzly pokial maju deti
        if len(database[index].get_child_all()) != 0:
            database = database[index].get_child_all()
            # dosiahli sme leaf node
            # uzol uz deti nema, ale moze mat viac ako jeden descriptor
            # toto je z toho dovodu ze pri tvorbe stromu som sa snazil
            # aby bolo najmenej 10 KVP na uzol, no niekedy je ich aj menej...
            # nechcem to uz riesit necham to tak
        else:
            lowest = sys.float_info.max

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
                break

    return most_similar_object


# DEBUG
def db_size(db):
    global DB_SIZE

    if len(db) == 0:
        return

    for item in db:
        DB_SIZE += 1
        if item is None:
            continue
        db_size(item.get_child_all())


# DEBUG
def db_unique_items(db):
    global DB_UNIQUE_ITEMS

    if len(db) == 0:
        return

    for item in db:
        if item is None:
            continue

        for path in item.get_kvps().get_key():
            DB_UNIQUE_ITEMS.append(path)

        db_unique_items(item.get_child_all())

# starting point
if __name__ == '__main__':

    global DB_SIZE

    start = time.time()

    database = create_tree()

    print("Tree creation took ", time.time() - start, 's')

    descriptors = get_query_image_descriptors()

    start = time.time()

    for desc in descriptors:
        mark_node_visits_in_db(database, desc)

    print("Marking visited nodes done in", time.time() - start ,'s')

    db_copy = deepcopy(database)

    start = time.time()

    create_query_tree(db_copy)

    print("Query tree created in ", time.time() - start, 's')

    # debug
    db_size(database)
    print("ORIG DB: " + str(DB_SIZE))

    # debug
    DB_SIZE = 0
    db_size(db_copy)
    print("COPY DB: " + str(DB_SIZE))

    # debug
    db_unique_items(db_copy)
    print(Counter(DB_UNIQUE_ITEMS).keys())  # equals to list(set(words))
    print(Counter(DB_UNIQUE_ITEMS).values())  # counts the elements' frequency

    print("DONE")
