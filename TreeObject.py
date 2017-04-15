from KeyValuePair import KeyValuePair
from KeyValuePairList import KeyValuePairList
import numpy as np
import math
import cv2


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
        self.kvp_n_of_desc_of_img = [] # num of descriptors of images separately


        self.w = 0 # = w, weight
        self.m = 0 # = m, num of all descriptors, of all images combined
        self.d = 0 # = d = m * w
        self.q = 0 # = q = n * w

        # LEN pri query strome.
        # Pocet deskriptorov ktore presli tymto uzlom
        # malo by korespondovat s touto vetou v clanku:
        # ni is the number of descriptor vectors of the
        # query with a path through
        # node i
        self.n = 0

        self.q_vector = np.empty((0, 3), dtype=np.float)
        self.d_vector = np.empty((0, 3), dtype=np.float)

        self.parent = None

        # relevance score S
        self.s = 0.0

        # IMG KVP = Vektor ktory obsahuje pocty descriptorov
        # daneho obrazku po ceste stromom dolu
        self.vector_img_kvp_counts = []

        # vektor ktory obsahuje pocty query descriptorov
        # ktore navstivili dany uzol, pozri query setup
        self.vector_query_visits_counts = []

        if children is not None:
            for child in children:
                self.add_child(child)

    def get_kvps(self):
        return self.kvps

    def get_label(self):
        return self.label

    def get_child(self, index):
        return self.children[index]

    def get_child_all(self):
        return self.children

    def kill_children(self):
        self.children = []

    def get_weight(self):
        return self.w

    def add_child(self, node, parent=None):
        assert isinstance(node, TreeObject)
        self.children.append(node)
        if parent is not None:
            node.parent = parent

    def visit(self):
        self.n += 1

    def get_num_of_visits(self):
        return self.n

    def get_center(self):
        return self.center

    def setup(self, num_of_pics):

        # kazdy uzol vie pre kazdy obrazok pocet vsetkych deskriptorov toho obrazku
        paths, counts = np.asarray(np.unique(self.kvps.get_key(), return_counts=True))
        for i in range(0, len(paths)):
            self.kvp_n_of_desc_of_img.append(KeyValuePair(paths[i], counts[i]))

        # kazdy uzol bude mat vahu w = ln(pocet_vsetkych_obrazkov_v_databaze / pocet_obrazkov_ktore_obsahuju_dany_deskriptor)
        # pocet_obrazkov_ktore_obsahuju_dany_deskriptor = malo by to odpovedat pocet obrazkov asociovanych s danym uzlom
        # pocet kvp = pocet obrazkov, kedze je to roztriedene
        self.w = math.log(num_of_pics / len(self.kvp_n_of_desc_of_img))

        # kazdy uzol ma m (rovne suctu m kazdeho obrazka) (pocet vsetkych desc)
        self.m = len(self.kvps.get_value())

        # na kazdom uzle je mozne po vytvoreni stromu predpocitat d = m * w
        # TODO: m = dufam ze to 'm' ktore je rovne suctu m kazdeho obrazka
        self.d = self.m * self.w

        if self.parent is None:
            for item in self.kvp_n_of_desc_of_img:
                self.vector_img_kvp_counts.append(KeyValuePairList(item.get_key(), item.get_value()))

    def get_q_vector(self):
        return self.q_vector

    def get_d_vector(self):
        return self.d_vector

    def query_setup(self):
        self.q = self.n * self.w

        self.q_vector = np.append(self.q_vector, self.q)
        if self.parent is not None:
            for item in self.parent.get_q_vector():
                self.q_vector = np.append(self.q_vector, item)

        self.d_vector = np.append(self.d_vector, self.d)
        if self.parent is not None:
            for item in self.parent.get_d_vector():
                self.d_vector = np.append(self.d_vector, item)

        self.normalize(self.q_vector)
        self.normalize(self.d_vector)

        # TODO: naco to vobec pocitam ked potom robim aj tak s poctami descriptorov
        # ktore normalizujem?
        self.compute_relevance_score()

        if self.parent is not None:
            for item in self.kvp_n_of_desc_of_img:
                self.vector_img_kvp_counts.append(KeyValuePairList(item.get_key(), item.get_value()))

            for item in self.parent.vector_img_kvp_counts:
                for kvpl in self.vector_img_kvp_counts:
                    if kvpl.get_key() == item.get_key():
                        kvpl.add_value(item.get_values())

        self.vector_query_visits_counts.append(self.n)

        if self.parent is not None:
            for item in self.parent.vector_query_visits_counts:
                self.vector_query_visits_counts.append(item)

    @staticmethod
    def normalize(vector):

        pom = 0.0
        for i in vector:
            pom += i * i

        length = abs(math.sqrt(pom))

        for idx,_ in enumerate(vector):
            vector[idx] = vector[idx]/length

    def compute_relevance_score(self):
        if self.q_vector is None or len(self.q_vector) == 0:
            print("Missing Q values")
            return

        if self.d_vector is None or len(self.d_vector) == 0:
            print("Missing D values")
            return

        if len(self.d_vector) != len(self.q_vector):
            print("Q and D lists different lenghts")
            return

        pom = 0.0

        for i in range(len(self.q_vector)-1):
            pom += self.q_vector[i]*self.d_vector[i]
        # TODO: aky ma zmysel skore ked ratam len pocty deskriptorov?
        self.s = 1.0 - pom

    @staticmethod
    def compute_relevance_score_vztah6(vector1, vector2):
        pom = 0.0

        for i in range(len(vector1)):
            pom += vector1[i] * vector2[i]

        return 1.0-pom

    @staticmethod
    def compute_relevance_score_vztah3(vector1, vector2):
        result = np.subtract(vector1, vector2)
        pom = 0.0
        for i in result:
            pom += i

        return 1-pom
