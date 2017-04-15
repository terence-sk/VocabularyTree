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

    def set_value(self, value):
        self.value = value

    @staticmethod
    def get_value_by_key(kvpList, key):
        for k, v in kvpList:
            if k == key:
                return v

    @staticmethod
    def get_item_if_in_list(kvp_list, name):
        for item in kvp_list:
            if item.get_key() == name:
                return item
        return None
