from entity_id import EntityId


class IdHandler:

    def __init__(self):

        self.id_count = 1
        self.ids_pool = list()

    def create_next_id(self, frame_number):

        obj_id = EntityId(self.id_count, frame_number)
        self.id_count += 1
        self.ids_pool.append(obj_id)

        return obj_id

