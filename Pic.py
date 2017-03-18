class Pic:
    def __init__(self, pathToImg, img):
        self.path = pathToImg
        self.img = img

    def getpath(self):
        return self.path

    def getimg(self):
        return self.img