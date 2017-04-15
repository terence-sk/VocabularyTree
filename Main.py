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
import pickle
from PIL import Image

MINIMUM_KEYPOINTS = 10
NUM_OF_CLUSTERS = 10
NUM_OF_LEVELS = 3
NUM_OF_PICS = -1
QUERY_TREE = []

DB_UNIQUE_ITEMS = []

PICTURE_SCORES = []


def load_pics():

    current_dir = os.getcwd()

    pics = current_dir + "/pics"

    pic_list = []

    for filename in os.listdir(pics):
        img = cv2.imread(pics + "/" + filename, cv2.IMREAD_GRAYSCALE)
        pic_list.append(Pic(filename, img))
        PICTURE_SCORES.append(KeyValuePair(filename, 0))

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
    # branch factor je odporucany na 10

    # center = list of x arrays of length 128 - descriptors of cluster centers
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

    # Toto je root
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
    query_path = os.getcwd() + "/query_pic/query.jpg"
    query_img = cv2.imread(query_path, cv2.IMREAD_GRAYSCALE)
    kp, desc = get_keypoint_descriptors_tuple(query_img)

    print("We are looking for a picture named: ", query_path)

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


# Pokus c 1. nefunguje , len ak dam hladat taky obrazok ktory uz v db je
def get_result(query_tree):

    # prechadzam strom pokym sa nedostanem na taky uzol
    # ktory uz nema deti === leaf node, popritom ratam
    # kolko descriptorov mal po ceste query aj obrazok (tvorim vector)
    for node in query_tree:
        if node is None:
            continue

        if len(node.get_child_all()) != 0:
            get_result(node.get_child_all())
        else:
            values_vector = []
            keys_vector = []
            for kvp in node.kvp_n_of_desc_of_img:
                values_vector.append(kvp.get_value())
                keys_vector.append(kvp.get_key())

                values_vector = [int(i) for i in values_vector]

            TreeObject.normalize(values_vector)

            for key, norm_value in zip(keys_vector, values_vector):
                item = KeyValuePair.get_item_if_in_list(PICTURE_SCORES, key)
                if item is not None:
                    item.set_value(item.get_value() + norm_value)


# len pooznacuje visits v databazovych objektoch
def mark_node_visits_in_db(database, descriptor):
    most_similar_object = None
    while True:

        highest = sys.float_info.min
        index = -1

        # list 10tich TreeObjectov
        for idx, node in enumerate(database):
            # porovnavam jednotlive centra s deskriptorom
            # skalarny sucin(dot product), hlada najvacsiu hodnotu,
            # nie euclidean distance a najmensiu ako povodne
            dist = np.dot(node.get_center(), descriptor)
            if dist > highest:
                highest = dist
                index = idx

        database[index].visit()

        # prechadzame uzly pokial maju deti
        if len(database[index].get_child_all()) != 0:
            database = database[index].get_child_all()
            # dosiahli sme leaf node
            # uzol uz deti nema, ale moze mat viac ako jeden descriptor
            # toto je z toho dovodu ze pri tvorbe stromu som sa snazil
            # aby bolo najmenej 10 KVP na uzol, no niekedy je ich aj menej...
            # tuto dilemu riesi skorovanie, tu som to riesil len hladanim najblizsieho
            # descriptora k centru

        # --- IRELEVANTNY TEST, TOTO V ALGORITME ORIGINAL NIE JE  ---
        else:
            highest = sys.float_info.min

            # ak ostal uz len jeden KVP alebo su vsetky KVP z jedneho obrazku, mozme rovno vratit jeho nazov
            if len(database[index].get_kvps().get_key()) == 1 or len(np.unique(database[index].get_kvps().get_key())) == 1:
                return database[index].get_kvps().get_key()[0]

            # v opacnom pripade prejdeme este deskriptory leaf nodu
            else:
                leaf_node_descs = database[index].get_kvps()

                for idx, val in enumerate(leaf_node_descs.get_value()):
                    dist = np.dot(val, descriptor)

                    if dist > highest:
                        highest = dist
                        most_similar_object = leaf_node_descs.get_key()[idx]
                break

    return most_similar_object


if __name__ == '__main__':

# ---------------

    start = time.time()

    database = create_tree()

    print("Tree creation took ", time.time() - start, 's')
# ---------------

    descriptors = get_query_image_descriptors()

    start = time.time()

    for desc in descriptors:
        mark_node_visits_in_db(database, desc)

    print("Marking visited nodes done in", time.time() - start ,'s')

# ---------------

    db_copy = deepcopy(database)

    start = time.time()

    create_query_tree(db_copy)

    print("Query tree created in ", time.time() - start, 's')

# ---------------

    #compute_img_scores()

# ---------------
    start = time.time()

    get_result(db_copy)

    print("Result found in ", time.time() - start, 's')

    highest = 0
    name = None
    for kvp in PICTURE_SCORES:
        if kvp.get_value() > 0:
            print(kvp.get_key())
            print(kvp.get_value())
            print("---------")

            if kvp.get_value() > highest:
                highest = kvp.get_value()
                name = kvp.get_key()

    current_dir = os.getcwd()
    pics = current_dir + "/pics/"

    img = Image.open(pics+name)
    img.show()


    print("DONE")
