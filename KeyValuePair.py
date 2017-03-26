class KeyValuePair:

    # key = path img
    # value = descriptor / pocet descriptorov, zalezi od pouzitia
    def __init__(self, key, value):
        self.key = key
        self.value = value

    def get_key(self):
        return self.key

    def get_value(self):
        return self.value