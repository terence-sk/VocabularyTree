import os
import cv2
import numpy as np
import sys
import operator
import time
from copy import deepcopy
from PIL import Image
from KeyValuePair import KeyValuePair
from TreeObject import TreeObject
from Pic import Pic

VISITS = 0
VISIT_DESCS = 0
NUM_OF_CLUSTERS = 10
NUM_OF_PICS = -1

PICTURE_SCORES = []
LEAF_NODES = []
DB_SIZE = 0


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

    # center = list of x arrays of length 128 - descriptors of cluster centers
    # compactness = It is the sum of squared distance from each point to their corresponding centers.
    # label = Label of keypoint to which cluster does keypoint belong
    # K = Branch factor in this case
    compactness, labels, centers = cv2.kmeans(data=descs, K=NUM_OF_CLUSTERS, bestLabels=None, criteria=criteria, attempts=10,
                                      flags=cv2.KMEANS_RANDOM_CENTERS)

    return compactness, labels, centers


def tree_add_node(item):

    item.setup(NUM_OF_PICS)

    if len(item.get_kvps().get_value()) <= 10:
        return

    compactness, labels, center = get_clustered_data(item.get_kvps().get_value())

    descriptors = item.get_kvps().get_value()
    paths = item.get_kvps().get_key()

    for i in range(0, NUM_OF_CLUSTERS):

        desc = descriptors[labels.ravel() == i]
        path = paths[labels.ravel() == i]
        kvp = KeyValuePair(path, desc)

        item.add_child(TreeObject(kvp, compactness=compactness, label=i, center=center[i]), parent=item)

        tree_add_node(item.get_child(i))


def create_tree():

    # Toto je root
    tree = []
    descriptors = []
    paths = []

    start = time.time()

    pics = load_pics()

    print('Pics loaded in ' ,time.time() - start, 's')

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
        else:
            break


def get_leaf_nodes(tree):
    if len(tree) == 0:
        return

    for item in tree:
        if item is None:
            continue
        if len(item.get_child_all()) == 0:
            LEAF_NODES.append(item)
        else:
            tree = item.get_child_all()
            get_leaf_nodes(tree)


def compute_img_scores(tree):
    get_leaf_nodes(tree)
    for item in LEAF_NODES:

        TreeObject.normalize(item.vector_query_visits_counts)
        TreeObject.normalize(item.vector_img_kvp_counts[0].get_values())

        for img in item.vector_img_kvp_counts:
            TreeObject.normalize(img.get_values())

            for pic in PICTURE_SCORES:
                if pic.get_key() == img.get_key():
                    pic.set_value(pic.get_value() + TreeObject.compute_relevance_score_vztah3(item.vector_query_visits_counts, img.get_values()))
                    #pic.set_value(pic.get_value() + TreeObject.compute_relevance_score_vztah6(item.vector_query_visits_counts,img.get_values()))

    my_dict = {}
    for pic in PICTURE_SCORES:
        my_dict[pic.get_key()] = pic.get_value()

    my_dict = (sorted(my_dict.items(), key=operator.itemgetter(1)))

    for key in my_dict:
        print(key[0], ' scored ', key[1])


def db_size(db):
    global DB_SIZE

    if len(db) == 0:
        return

    for item in db:
        DB_SIZE += 1
        if item is None:
            continue
        db_size(item.get_child_all())


def bruteforce(database, descriptor):
    global VISITS
    global VISIT_DESCS
    for item in database:
        VISITS += 1
        for desc in item.get_kvps().get_value():
            VISIT_DESCS += 1
            if np.allclose(desc, descriptor, atol=62):
                print("BF FOUND:", item.get_kvps().get_key())
        if len(item.get_child_all()) != 0:
            database = item.get_child_all()
            bruteforce(database, descriptor)



if __name__ == '__main__':

# ---------------

    start = time.time()

    database = create_tree()

    print("Tree creation took ", time.time() - start, 's')
# ---------------

    descriptors = get_query_image_descriptors()

    db_size(database)
    print("velkost db ", DB_SIZE)

    DB_SIZE = 0

    start = time.time()

    bruteforce(database, descriptors[0])
    print("pocet navstev nodov: ", VISITS)
    print("pocet navstev descs: ", VISIT_DESCS)

    print("Bruteforce done in: ", time.time() - start)

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
    start = time.time()
    # sposob1, zly
    #get_result(db_copy)

    # sposob2
    compute_img_scores(db_copy)

    db_size(db_copy)
    print(DB_SIZE)

    print("Result found in ", time.time() - start, 's')

    highest = 0
    name = None
    for kvp in PICTURE_SCORES:
        if kvp.get_value() > highest:
            highest = kvp.get_value()
            name = kvp.get_key()

    print(name, highest)

    current_dir = os.getcwd()
    pics = current_dir + "/pics/"

    img = Image.open(pics+name)
    img.show()


    print("DONE")
