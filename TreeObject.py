class TreeObject:

    def __init__(self, pathToImg, descriptors, children=None):
        self.imgPath = pathToImg
        self.desc = descriptors
        self.children = []
        if children is not None:
            for child in children:
                self.add_child(child)

    def getDescriptor(self):
        return self.desc

    def getImgPath(self):
        return self.imgPath

    def add_child(self, node):
        assert isinstance(node, TreeObject)
        self.children.append(node)
