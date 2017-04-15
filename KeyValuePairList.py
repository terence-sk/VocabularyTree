class KeyValuePairList:

    # key = path img
    # value = descriptor / pocet descriptorov, zalezi od pouzitia
    def __init__(self, key, value):
        self.key = key
        self.value = []

        if value is not None:
            self.value.append(int(value))

    def get_key(self):
        return self.key

    def get_values(self):
        return self.value

    def add_value(self, value):
        if isinstance(value, list):
            for item in value:
                self.value.append(int(item))
        else:
            self.value.append(int(value))
