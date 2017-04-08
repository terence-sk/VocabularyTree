class KeyValuePair:

    # key = path img
    # value = descriptor / pocet descriptorov, zalezi od pouzitia
    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.visit = [0] * len(key)

    def get_key(self):
        return self.key

    def get_value(self):
        return self.value

    def get_visit_array(self):
        return self.visit