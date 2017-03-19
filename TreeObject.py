class TreeObject:

    # KVP = Key Value Pair = Descriptor + path
    # KVP By malo byt pole, pretoze urcite bude jeden objekt
    # obsahovat viacero deskriptorov

    # kvps = Hodnoty (deskriptori s cestami k obrazkom)
    # label = label pridelena algoritmom kmeans
    # children = deti objektu, takisto TreeObjecty
    def __init__(self, kvps, label, children=None):
        self.kvps = kvps
        self.children = []
        self.label = label

        if children is not None:
            for child in children:
                self.add_child(child)

    def getkvps(self):
        return self.kvps

    def getlabel(self):
        return self.label

    def add_child(self, node):
        assert isinstance(node, TreeObject)
        self.children.append(node)

    def get_child(self, index):
        return self.children[index]
