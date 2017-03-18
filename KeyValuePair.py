class KeyValuePair:

    # key = path
    # value = descriptor
    def __init__(self, key, value):
        self.key = key
        self.value = value

    def getkey(self):
        return self.key

    def getvalue(self):
        return self.value