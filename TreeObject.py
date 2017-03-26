from KeyValuePair import KeyValuePair
import numpy as np
import math


class TreeObject:

    # KVP = Key Value Pair = Descriptor + path
    # KVP By malo byt pole, pretoze urcite bude jeden objekt
    # obsahovat viacero deskriptorov

    # center = SIFT 128, centrum clustra tohto uzla
    # compactness = sum of squared distance from each point to their corresponding centers
    # kvps = LIST Hodnoty (deskriptori s cestami k obrazkom)
    # label = INT label pridelena algoritmom kmeans
    # children = LIST deti objektu, takisto TreeObjecty
    # kvp_n_of_desc_of_img = LIST pocet descriptorov jednotlivych obrazkov
    # weight = vaha daneho uzlu vyratana algoritmom v setupe

    def __init__(self, kvps, compactness, label, center, children=None):

        self.kvps = kvps

        self.compactness = compactness
        self.label = label
        self.center = center

        self.children = []
        self.kvp_n_of_desc_of_img = []
        self.weight = -1
        self.num_of_all_desc = -1

        if children is not None:
            for child in children:
                self.add_child(child)

    def get_kvps(self):
        return self.kvps

    def get_label(self):
        return self.label

    def get_child(self, index):
        return self.children[index]

    def get_weight(self):
        return self.weight

    def add_child(self, node):
        assert isinstance(node, TreeObject)
        self.children.append(node)

    def setup(self, num_of_pics):

        # kazdy uzol vie pre kazdy obrazok pocet vsetkych deskriptorov toho obrazku
        paths, counts = np.asarray(np.unique(self.kvps.get_key(), return_counts=True))

        for i in range(0, len(paths)):
            self.kvp_n_of_desc_of_img.append(KeyValuePair(paths[i], counts[i]))

        # kazdy uzol bude mat vahu w = ln(pocet_vsetkych_obrazkov_v_databaze / pocet_obrazkov_ktore_obsahuju_dany_deskriptor)
        # malo by to odpovedat pocet obrazkov asociovanych s danym uzlom
        # pocet kvp = pocet obrazkov, kedze je to roztriedene
        self.weight = math.log(num_of_pics / len(self.kvp_n_of_desc_of_img))

        # kazdy uzol ma m (rovne suctu m kazdeho obrazka) (pocet vsetkych desc)
        self.num_of_all_desc = len(self.kvps.get_value())